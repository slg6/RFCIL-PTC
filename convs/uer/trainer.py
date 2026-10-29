import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from tqdm import tqdm

def load_or_initialize_parameters(args, model):
    try:
        # 尝试加载预训练模型的权重
        pretrained_weights = torch.load(args.pretrained_model_path)
        model_dict = model.state_dict() #你可以直接修改 state_dict 中的张量，然后使用 load_state_dict() 将修改后的权重加载到模型中。
        # 过滤掉不匹配的权重
        filtered_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 更新模型的权重
        model_dict.update(filtered_weights)
        model.load_state_dict(model_dict)
        print("Successfully loaded pre-trained model weights.")
    except Exception as e:
        print(f"Failed to load pre-trained model weights: {e}")
        print("Initializing model parameters randomly.")
        # 如果加载失败，可以随机初始化模型参数
        model.apply(model._init_weights)
def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    print("Load vocabulary.")
    if args.spm_model_path:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                              "pip install sentencepiece")
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.spm_model_path)
        args.vocab = {sp_model.IdToPiece(i): i for i
                      in range(sp_model.GetPieceSize())}
        args.tokenizer = str2tokenizer[args.tokenizer](args)    
        if args.target == "seq2seq":
            tgt_sp_model = spm.SentencePieceProcessor()
            tgt_sp_model.Load(args.tgt_spm_model_path)
            args.tgt_vocab = {tgt_sp_model.IdToPiece(i): i for i
                              in range(tgt_sp_model.GetPieceSize())}
    else:
        args.tokenizer = str2tokenizer[args.tokenizer](args)    #BertTokenizer(args)
        args.vocab = args.tokenizer.vocab
        if args.target == "seq2seq":
            tgt_vocab = Vocab()
            tgt_vocab.load(args.tgt_vocab_path)
            args.tgt_vocab = tgt_vocab.w2i
    #slg
    # train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)
    # train_loader = iter(train_loader)
    # flag = 0
    # for i in next(train_loader):
    #     flag += 1
    # print(flag)
    #slg
    # Build model.
    print("Build model.")
    model = build_model(args)   #创建模型结构
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())
    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        # model = load_model(model, args.pretrained_model_path) #加载预训练模型
        load_or_initialize_parameters(args, model)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    if args.dist_train:
        # Multiprocessing distributed mode.
        print("Multiprocessing distributed mode.")
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        print("Single GPU mode.")
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        print("CPU mode.")
        worker(None, None, args, model)


class Trainer(object):
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps       #accumulation_steps 参数用于实现梯度累积，通过在多个小批次上累积梯度，模拟大批次训练的效果。这种方法可以在不增加硬件负担的情况下提高模型的训练效果，尤其适用于内存受限的场景。
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps

        self.output_model_path = args.output_model_path

        self.start_time = time.time()
        self.total_loss = 0.0
        
        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size

    def forward_propagation(self, batch, model):

        raise NotImplementedError

    def report_and_reset_stats(self):

        raise NotImplementedError

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        model.train()
        loader_iter = iter(loader)
        
        while True:
            if self.current_step == self.total_steps + 1:       #每一步仅训练一个batch_size，不是所有的数据
                break
            batch = list(next(loader_iter))     #先加载22560个实例，一次从中取出一个batch的实例，取完以后，再补充。 一个batch实例含有batch_size个样本
            self.seq_length = batch[0].size(1)
            if gpu_id is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(gpu_id)

            loss = self.forward_propagation(batch, model)

            if args.fp16:
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.current_step % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            # print(f"self.current_step % self.report_steps, {self.current_step} % {self.report_steps}")
            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                
                self.report_and_reset_stats()
                self.start_time = time.time()

            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                save_model(model, self.output_model_path + "-" + str(self.current_step))    #保存中间模型

            self.current_step += 1


class MlmTrainer(Trainer):
    def __init__(self, args):
        super(MlmTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0
        
class BertPacketsTrainer(Trainer):
    def __init__(self, args):
        super(BertPacketsTrainer, self).__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0

        # self.total_loss += loss.item()
        # self.total_loss_mlm += loss_mlm.item()
        # self.total_loss_sp += loss_sp.item()
        # self.total_loss_fm += loss_fm.item()
        # self.total_correct_mlm += correct_mlm.item()
        # self.total_correct_sp += correct_sp.item()
        # self.total_correct_fm += correct_fm.item()
        # self.total_denominator += denominator_mlm.item()
        # self.total_denominator_fm += denominator_fm.item()
        # self.total_instances += src.size(0)     #总实例，与batch_size一样

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_loss_fm = 0.0
        self.total_correct_fm = 0.0
        self.total_denominator = 0.0
        self.total_denominator_fm = 0.0
        self.load_balance_alpha = args.moebert_load_balance
        self.is_moe = args.is_moe
        self.log_file = args.log_file

    def forward_propagation(self, batch, model):
        debug_mode = False
        if debug_mode:
            print("In function forward_propagation(self, batch, model):")
            print("type of batch:",type(batch))
            print("type of the content of batch:",[type(elem) for elem in batch])
            '''
            具体实例：
            instance = (
                [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src
                [(3, 2129), (6, 2003)],                              #tgt_mlm 3，6被掩码  掩码前值为2129，2003
                0,                                                    #label  0 表示句子对是连续的。  1 表示句子对是随机拼接的。  修改成数据包位置
                [(5, 2120), (21, 9)],                              #field_modified 5，21被随机替换  替换前值为2120，9
                0                                                     #proto 协议为TCP
            )
            '''
            #loss_info是model的forward函数计算出来的  
            #传给target层的是（含有CLS的张量，tgt）   tgt= (tgt_mlm, tgt_sp)   tgt_mlm是用于字词预测的不用修改
            #修改tgt_sp，代表包的位置即可。    只需要修改数据的传入方式即可
            '''
            model
            Pretraining models consist of three parts:
                - embedding
                - encoder
                - target
            ''' 
        if len(batch)==6:
            src, tgt_mlm, label, field_modified, seg, proto = batch
                    # 在 embeddings.py 里 forward 前临时加
            loss_info = model(src, (tgt_mlm, label, field_modified), seg,proto)
        # else:
        #     src, tgt_mlm, tgt_sp, seg, tgt_pos = batch
        #     loss_info = model(src, (tgt_mlm, tgt_sp, tgt_pos), seg)
        
        if self.is_moe:
            loss_mlm, loss_sp, loss_fm, correct_mlm, correct_sp, correct_fm,denominator_mlm, denominator_fm, gate_loss = loss_info
            # loss_mlm, loss_msp, loss_fm,correct_mlm, correct_msp, correct_fm, denominator_mlm, denominator_fm
        else:
            loss_mlm, loss_sp, loss_fm,correct_mlm, correct_sp, correct_fm, denominator_mlm, denominator_fm = loss_info
            gate_loss = 0.0
        loss = loss_mlm/10 + loss_sp + loss_fm/10 + self.load_balance_alpha * gate_loss  #MLM损失   序列损失  包位置损失  MOE下的平衡 （此处的权重是不是能修改一下）
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_loss_fm += loss_fm.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_correct_fm += correct_fm.item()
        self.total_denominator += denominator_mlm.item()
        self.total_denominator_fm += denominator_fm.item()
        self.total_instances += src.size(0)     #总实例，与batch_size一样
        loss = loss / self.accumulation_steps

        return loss
    def report_and_reset_stats(self):
        import logging
        import time
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler(self.log_file), #表示log写入文件
                                logging.StreamHandler()     #表示log打印终端
                            ])
        logger = logging.getLogger(__name__)
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        log_message = (
        f"| {self.current_step:8d}/{self.total_steps:8d} steps"
        f"| {time.time() - self.start_time:3.3f} s"
        f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s"
        f"| loss {self.total_loss / self.report_steps:7.2f}"
        f"| loss_mlm: {self.total_loss_mlm / self.report_steps:3.3f}"
        f"| loss_sp: {self.total_loss_sp / self.report_steps:3.3f}"
        f"| loss_fm: {self.total_loss_fm / self.report_steps:3.3f}"
        f"| acc_mlm: {self.total_correct_mlm / self.total_denominator:3.3f}"
        f"| acc_sp: {self.total_correct_sp / self.total_instances:3.3f}"
        f"| acc_fm: {self.total_correct_fm / self.total_denominator_fm:3.3f}"

    )
        logger.info(log_message)
        # print("| {:8d}/{:8d} steps"
        #       "| {:3.3f} s"
        #       "| {:8.2f} tokens/s"
        #       "| loss {:7.2f}"
        #       "| loss_mlm: {:3.3f}"
        #       "| loss_sp: {:3.3f}"
        #       "| loss_pos: {:3.3f}"
        #       "| acc_mlm: {:3.3f}"
        #       "| acc_sp: {:3.3f}"
        #       "| acc_pos: {:3.3f}".format(
        #     self.current_step,
        #     self.total_steps,
        #     (time.time() - self.start_time),
        #     done_tokens / (time.time() - self.start_time),
        #     self.total_loss / self.report_steps,
        #     self.total_loss_mlm / self.report_steps,
        #     self.total_loss_sp / self.report_steps,
        #     self.total_loss_pos / self.report_steps,
        #     self.total_correct_mlm / self.total_denominator,
        #     self.total_correct_sp / self.total_instances,
        #     self.total_correct_pos / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp, self.total_loss_fm = 0.0, 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator, self.total_denominator_fm = 0.0, 0.0, 0.0
        self.total_correct_sp, self.total_correct_fm, self.total_instances = 0.0, 0.0, 0.0

class BertPacketTrainer(Trainer):
    def __init__(self, args):
        super(BertPacketTrainer, self).__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0
        
        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_loss_pos = 0.0
        self.total_correct_pos = 0.0
        self.total_denominator = 0.0
        self.load_balance_alpha = args.moebert_load_balance
        self.is_moe = args.is_moe
        self.log_file = args.log_file

    def forward_propagation(self, batch, model):
        debug_mode = False
        if debug_mode:
            print("In function forward_propagation(self, batch, model):")
            print("type of batch:",type(batch))
            print("type of the content of batch:",[type(elem) for elem in batch])
            '''
            batch具体实例：
            instance = (
                [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src 输入序列（Token ID 列表）。
                [(3, 2129), (6, 2003)],                              #tgt_mlm 3，6被掩码  掩码前值为2129，2003
                0,                                                    #tgt_sp  0 表示句子对是连续的。  1 表示句子对是随机拼接的。
                [4, 8],                                              #seg  Segment A 结束于索引 4，Segment B 结束于索引 8
                0,                                                     #A包位置
                0                                                     #proto 协议为TCP
            )

            '''
            #loss_info是model的forward函数计算出来的  
            #传给target层的是（含有CLS的张量，tgt）   tgt= (tgt_mlm, tgt_sp)   tgt_mlm是用于字词预测的不用修改
            #修改tgt_sp，代表包的位置即可。    只需要修改数据的传入方式即可
            '''
            model
            Pretraining models consist of three parts:
                - embedding
                - encoder
                - target
            ''' 
        if len(batch)==6:
            src, tgt_mlm, tgt_sp, seg, tgt_pos, proto = batch
            loss_info = model(src, (tgt_mlm, tgt_sp, tgt_pos), seg ,proto)
        else:
            src, tgt_mlm, tgt_sp, seg, tgt_pos = batch
            loss_info = model(src, (tgt_mlm, tgt_sp, tgt_pos), seg)
        
        if self.is_moe:
            loss_mlm, loss_sp, loss_pos, correct_mlm, correct_sp, correct_pos,denominator, gate_loss = loss_info
        else:
            loss_mlm, loss_sp, loss_pos,correct_mlm, correct_sp, correct_pos,denominator = loss_info
            gate_loss = 0.0
        loss = loss_mlm/10 + loss_sp + loss_pos + self.load_balance_alpha * gate_loss  #MLM损失   序列损失  包位置损失  MOE下的平衡 （此处的权重是不是能修改一下）
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_loss_pos += loss_pos.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_correct_pos += correct_pos.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)     #总实例，与batch_size一样
        loss = loss / self.accumulation_steps

        return loss
    def report_and_reset_stats(self):
        import logging
        import time
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler(self.log_file),
                                logging.StreamHandler()
                            ])
        logger = logging.getLogger(__name__)
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        log_message = (
        f"| {self.current_step:8d}/{self.total_steps:8d} steps"
        f"| {time.time() - self.start_time:3.3f} s"
        f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s"
        f"| loss {self.total_loss / self.report_steps:7.2f}"
        f"| loss_mlm: {self.total_loss_mlm / self.report_steps:3.3f}"
        f"| loss_sp: {self.total_loss_sp / self.report_steps:3.3f}"
        f"| loss_pos: {self.total_loss_pos / self.report_steps:3.3f}"
        f"| acc_mlm: {self.total_correct_mlm / self.total_denominator:3.3f}"
        f"| acc_sp: {self.total_correct_sp / self.total_instances:3.3f}"
        f"| acc_pos: {self.total_correct_pos / self.total_instances:3.3f}"
    )
        logger.info(log_message)
        # print("| {:8d}/{:8d} steps"
        #       "| {:3.3f} s"
        #       "| {:8.2f} tokens/s"
        #       "| loss {:7.2f}"
        #       "| loss_mlm: {:3.3f}"
        #       "| loss_sp: {:3.3f}"
        #       "| loss_pos: {:3.3f}"
        #       "| acc_mlm: {:3.3f}"
        #       "| acc_sp: {:3.3f}"
        #       "| acc_pos: {:3.3f}".format(
        #     self.current_step,
        #     self.total_steps,
        #     (time.time() - self.start_time),
        #     done_tokens / (time.time() - self.start_time),
        #     self.total_loss / self.report_steps,
        #     self.total_loss_mlm / self.report_steps,
        #     self.total_loss_sp / self.report_steps,
        #     self.total_loss_pos / self.report_steps,
        #     self.total_correct_mlm / self.total_denominator,
        #     self.total_correct_sp / self.total_instances,
        #     self.total_correct_pos / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp, self.total_loss_pos = 0.0, 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_sp, self.total_correct_pos, self.total_instances = 0.0, 0.0, 0.0
    
class BertTrainer(Trainer):
    def __init__(self, args):
        super(BertTrainer, self).__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0
        self.load_balance_alpha = args.moebert_load_balance
        self.is_moe = args.is_moe

    def forward_propagation(self, batch, model):
        debug_mode = False
        if debug_mode:
            print("In function forward_propagation(self, batch, model):")
            print("type of batch:",type(batch))
            print("type of the content of batch:",[type(elem) for elem in batch])
        if len(batch)==5:
            src, tgt_mlm, tgt_sp, seg, proto = batch
            '''
            batch具体实例：
            instance = (
                [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src 输入序列（Token ID 列表）。
                [(3, 2129), (6, 2003)],                              #tgt_mlm 3，6被掩码  掩码前值为2129，2003
                0,                                                    #tgt_sp  0 表示句子对是连续的。  1 表示句子对是随机拼接的。
                [4, 8],                                              #seg  Segment A 结束于索引 4，Segment B 结束于索引 8
                0,                                                     #A包
                0                                                     #proto 协议为TCP
            )

            '''
            loss_info = model(src, (tgt_mlm, tgt_sp), seg, proto)  #loss_info是model的forward函数计算出来的  
            #传给target层的是（含有CLS的张量，tgt）   tgt= (tgt_mlm, tgt_sp)   tgt_mlm是用于字词预测的不用修改
            #修改tgt_sp，代表包的位置即可。    只需要修改数据的传入方式即可
            '''
            model
            Pretraining models consist of three parts:
                - embedding
                - encoder
                - target
            ''' 
        else:
            src, tgt_mlm, tgt_sp, seg = batch
            loss_info = model(src, (tgt_mlm, tgt_sp), seg)
        
        if self.is_moe:
            loss_mlm, loss_sp, correct_mlm, correct_sp, denominator, gate_loss = loss_info
        else:
            loss_mlm, loss_sp, correct_mlm, correct_sp, denominator = loss_info
            gate_loss = 0.0
        loss = loss_mlm/10 + loss_sp + self.load_balance_alpha * gate_loss  #MLM损失   序列损失   MOE下的平衡
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print("| {:8d}/{:8d} steps"
              "| {:3.3f} s"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_mlm: {:3.3f}"
              "| loss_sp: {:3.3f}"
              "| acc_mlm: {:3.3f}"
              "| acc_sp: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            (time.time() - self.start_time),
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_loss_mlm / self.report_steps,
            self.total_loss_sp / self.report_steps,
            self.total_correct_mlm / self.total_denominator,
            self.total_correct_sp / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_sp, self.total_instances = 0.0, 0.0


class AlbertTrainer(BertTrainer):
    pass


class LmTrainer(MlmTrainer):
    pass


class BilmTrainer(Trainer):
    def __init__(self, args):
        super(BilmTrainer, self).__init__(args)
        self.total_loss_forward, self.total_loss_backward = 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward = 0.0, 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_forward, tgt_backward, seg = batch
        loss_info = model(src, (tgt_forward, tgt_backward), seg)
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = loss_info
        loss = loss_forward + loss_backward
        self.total_loss += loss.item()
        self.total_loss_forward += loss_forward.item()
        self.total_loss_backward += loss_backward.item()
        self.total_correct_forward += correct_forward.item()
        self.total_correct_backward += correct_backward.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_forward {:3.3f}"
                  "| loss_backward {:3.3f}"
                  "| acc_forward: {:3.3f}"
                  "| acc_backward: {:3.3f}".format(
                    self.current_step,
                    self.total_steps, 
                    done_tokens / (time.time() - self.start_time), 
                    self.total_loss / self.report_steps,
                    self.total_loss_forward / self.report_steps,
                    self.total_loss_backward / self.report_steps,
                    self.total_correct_forward / self.total_denominator,
                    self.total_correct_backward / self.total_denominator))

        self.total_loss, self.total_loss_forward, self.total_loss_backward = 0.0, 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward, self.total_denominator = 0.0, 0.0, 0.0


class ClsTrainer(Trainer):
    def __init__(self, args):
        super(ClsTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_instances = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_instances))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0


class Seq2seqTrainer(Trainer):
    def __init__(self, args):
        super(Seq2seqTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_in, tgt_out, seg = batch
        loss_info = model(src, (tgt_in, tgt_out, src), seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()

        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class T5Trainer(Seq2seqTrainer):
    pass


class PrefixlmTrainer(MlmTrainer):
    pass


str2trainer = {"bert": BertTrainer, "bertflow": BertTrainer, "bertpacket":BertPacketTrainer,"bertpackets":BertPacketsTrainer,"mlm": MlmTrainer, "lm": LmTrainer,
               "albert": AlbertTrainer, "bilm": BilmTrainer, "cls": ClsTrainer,
               "seq2seq": Seq2seqTrainer, "t5": T5Trainer}

def worker(proc_id, gpu_ranks, args, model):    
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)
    
    print("Starting worker function")

    if args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None
    print("train loader constructing...")
    if args.dist_train:     #  if args.world_size > 1:   args.dist_train = True
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, rank, args.world_size, True)
    else:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)
    '''
    train_loader具体实例：
    instance = (            
        [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src
        [(3, 2129), (6, 2003)],                              # 3，6被掩码  掩码前值为2129，2003
        0,                                                    #  0 表示句子对是连续的。  1 表示句子对是随机拼接的。
        [4, 8],                                              # Segment A 结束于索引 4，Segment B 结束于索引 8
        0                                                     # 协议为TCP
    )

    train_loader修改目标：
    instance = (            
        [101, 7592, 103, 102, 2023, 103, 1034, 102, 0, 0, ...],  # src  一个数据包的内容
        [(3, 2129), (6, 2003)],                              # 3，6被掩码  掩码前值为2129，2003  不变
        0,                                                    #  0 表示句子对是连续的。  1 表示句子对是随机拼接的。0,1,2,3,4
        [4, 8],                                              #  token_A 结束于索引 4，tokne_B 结束于索引 8  
        0,                                                    #  0 表示A包在数据流中的位置
        0                                                     # 协议为TCP
    )

    '''
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    # Build optimizer.
    print("build optomizer...")
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps*args.warmup, args.total_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if args.dist_train:
        print("Initialize multiprocessing distributed training environment...")
        # Initialize multiprocessing distributed training environment.
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        # model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=False)
        print("Worker %d is training ... " % rank)
    else:
        print("Worker is training ...")

    trainer = str2trainer[args.target](args)    #BertTrainer(args)
    trainer.train(args, gpu_id, rank, train_loader, model, optimizer, scheduler)
