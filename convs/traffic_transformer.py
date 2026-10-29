import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(current_dir)
import argparse
from argparse import Namespace
import numpy as np
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.opts import finetune_opts

class TransformerAdapter(nn.Module):            #adapter 微调
    def __init__(self, hidden_size, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)

        #参考mos
        # self.activation = nn.GELU()
        self.activation = nn.ReLU()     

        #参考mos
        # self.dropout = nn.Dropout(dropout)
        self.dropout = dropout

        # 关键：初始化为近似恒等映射
        # nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.kaiming_uniform_(self.down_project.weight, a=math.sqrt(5)) #参考mos
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
        
    def forward(self, x):
        residual = self.down_project(x)
        residual = self.activation(residual)
        #参考mos 调整dropout层
        residual = nn.functional.dropout(residual, p=self.dropout, training=self.training)  

        # residual = self.up_project(residual)
        # residual = self.dropout(residual) 

        residual = self.up_project(residual)  #参考mos  *0.1
        return residual * 0.5

class TrafficTransformer(nn.Module):        #分类层就两个线性层，中间使用一个tanh激活函数进行连接。
    def __init__(self, args):
        args.is_moe = False
        super(TrafficTransformer, self).__init__()
        self.args = args
        # args.hidden_size = 768
        self.embedding = str2embedding['word_pos_seg'](args, len(args.tokenizer.vocab)) #词嵌入层 tokenID --> token向量  每个token都被转换为了一个向量，这个向量包含了：语义信息、位置信息、段落信息
        self.encoder = str2encoder['transformer'](args)  #编码器层 编码器层由多个Transformer层堆叠而成 生成最终表示向量  token向量 --> CLS
        # self.labels_num = labels_num   # 动态适配类别数
        self.pooling = "first"     # 池化方式，默认"first"取CLS
        self.soft_targets = False   #软标签参数（知识蒸馏相关）  # 是否使用软标签
        if "adapter_one" in self.args.model_name:
            pass
            # new_output_layer = nn.Linear(args.hidden_size, args.hidden_size)
            # self.output_layer_1 = nn.ModuleList()
            # self.output_layer_1.append(new_output_layer)
        else:
            self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dim = args.hidden_size

        self.adapters_mode = "normal"  # 控制所有 adapter 的模式
        self.adapter_groups = nn.ModuleList()

        if self.args.model_name == "tagfex":
            self.adapters_mode = "moe_adapters"

        if "adapter_one" in self.args.model_name:   #只添加一个adapter
            self.adapters_mode = "adapter_layer"
            self.adapter_layer = 11     #添加adapter的transformer层
            self.encoder.transformer[self.adapter_layer].adapters = nn.ModuleList()
            self.adapter_layer_num = None  #添加adapter的transformer层

        args = vars(args)

    def _add_new_adapter_layers(self, adapter_layer_num=1):
        '''
        为新任务添加一个adapter层
        adapter_layer_num:添加adapter的transformer层 默认为1层
        
        '''
        self.adapter_layer_num = adapter_layer_num
        bottleneck_dim = 256
        if "mgda" in self.args.model_name or "feature" in self.args.model_name:    #参考mos
            bottleneck_dim = 512
        new_adapters_list = nn.ModuleList()
        for i in range(adapter_layer_num):      #最后adapter_layer_num层添加adapter
            new_adapters = TransformerAdapter(self.args.hidden_size, bottleneck_dim=bottleneck_dim)
            new_adapters_list.append(new_adapters)
            if not hasattr(self.encoder.transformer[self.adapter_layer-(adapter_layer_num-1)+i], 'adapters'):    
                self.encoder.transformer[self.adapter_layer-(adapter_layer_num-1)+i].adapters = nn.ModuleList()
            self.encoder.transformer[self.adapter_layer-(adapter_layer_num-1)+i].adapters.append(new_adapters)
        
        self.adapter_groups.append(new_adapters_list)
        # new_output_layer = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        # self.output_layer_1.append(new_output_layer)    

    def _add_new_adapter_layer(self):
        '''
        为新任务添加一个adapter层
        '''
        bottleneck_dim = 256
        new_adapters = TransformerAdapter(self.args.hidden_size, bottleneck_dim=bottleneck_dim)
        self.adapter_groups.append(new_adapters)
        self.encoder.transformer[self.adapter_layer].adapters.append(new_adapters)

        new_output_layer = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.output_layer_1.append(new_output_layer)
    def _add_new_adapter_group(self):
        """为新任务添加一组 Adapter（每层一个）"""
        bottleneck_dim = 256
        if self.args.model_name == "tagfex":    #tagfex模式下太吃显存，显存不够用
            bottleneck_dim = 128
        if "mgda"   in self.args.model_name:    #参考mos
            bottleneck_dim = 16
        new_adapters = nn.ModuleList([
            TransformerAdapter(self.args.hidden_size, bottleneck_dim=bottleneck_dim) 
            for _ in range(self.args.layers_num)
        ])
        self.adapter_groups.append(new_adapters)
        # self.current_task_id = len(self.adapter_groups) - 1
        
        # 将新 Adapter 注册到对应的 TransformerLayer
        for i, adapter in enumerate(new_adapters):
            #这一层没有adapters就增加一个adapters模型列表
            if not hasattr(self.encoder.transformer[i], 'adapters'):    
                self.encoder.transformer[i].adapters = nn.ModuleList()
            #向这一层的adapters模型列表中添加一个adapter
            self.encoder.transformer[i].adapters.append(adapter)

    def switch(self, mode='normal'):
        """切换当前任务 Adapter 的模式"""
        self.adapters_mode = mode

    def forward(self, src, tgt=None, seg=None, soft_tgt=None,
                current_task_id=None, task_all=False, ):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
            current_task_id:TagFex算法当前增量任务
            task_all:TagFex算法是否一次完成已学任务的推理
        """
        # Embedding.
        seg = src[:,:,1].long()
        src = src[:,:,0].long()
        emb = self.embedding(src, seg)  #torch.Size([21, 320, 768])#完成了词嵌入、位置嵌入、段落嵌入。每个token都被转换为了一个向量，这个向量包含了：语义信息、位置信息、段落信息
        # Encoder.
        if self.adapters_mode == "normal":
            output = self.encoder(emb, seg)  # 直接通过编码器生成表示向量
        elif self.adapters_mode == "parallel_adapters":     #ssre算法 
            output = self.encoder(emb, seg, use_adapters=True)   # 通过编码器和 adapter 生成表示向量
        elif self.adapters_mode == "moe_adapters":      #tagfex算法
            if task_all:
                outputs = []
                for i in range(current_task_id+1):
                    output = self.encoder(emb, seg, moe_adapters=True, current_task_id=i)
                    outputs.append(output)
            else:
                output = self.encoder(emb, seg, moe_adapters=True, current_task_id=current_task_id) # 通过编码器和当前任务的 MoE adapter 生成表示向量
        elif self.adapters_mode == "adapter_one":   #此时output为列表
            outputs = self.encoder(emb, seg, adapter_layer=self.adapter_layer,
                                   adapter_layer_num=self.adapter_layer_num)
            outputs_output = []
            for i, output in enumerate(outputs):
                output = output[:, 0, :]
                # output = torch.tanh(self.output_layer_1[i](output)) 
                outputs_output.append(output)
            return {"features": outputs_output}

        # output = self.encoder(emb, seg) #生成强大的序列表示，尤其是 [CLS] token 的表示可以用于下游任务
        # fmaps = []

        if task_all: #TagFex算法
            result = []
            for i in range(len(outputs)):
                output = outputs[i]
                # seq_feature = output.permute(0, 2, 1).unsqueeze(-1)
                # fmaps = [seq_feature]
                output, fmaps = output
                output = output[:, 0, :]    # First.取第一个token的输出：即CLS  32*768
                output = torch.tanh(self.output_layer_1(output))
                result.append({'fmaps': fmaps,'features': output,})
            return result
        
        # seq_feature = output.permute(0, 2, 1).unsqueeze(-1)  # [batch_size x hidden_size x seq_length x 1] 32,768,401,1
        # fmaps = [seq_feature]
        output, fmaps = output

        output = output[:, 0, :]    # First.取第一个token的输出：即CLS  32*768
        # output = torch.tanh(self.output_layer_1(output))    #CLS先经过了一个线性层和tanh激活函数
        return {
            'fmaps': fmaps,
            'features': output,
        }
    
def traffic_transformer(pretrained=False, args=None):
    load_hyperparam(args)
    args1 = Namespace(**args)
    args1.tokenizer = str2tokenizer["bert"](args1)
    model = TrafficTransformer(args1)
    if pretrained:
        model.load_state_dict(torch.load
            ('~/llmfortraffic/IncrementalLearning/PyCIL/convs/models/pre-trained_model.bin/model_packet_dataset0_test.bin-120000', 
            map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'}), 
            strict=False
            )
    else:
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
    args['optimizer_scheduler'] = build_optimizer_scheduler(args1, model)
    return model


def read_dataset(args, path):   #读取数据集并将其转换为模型可处理的格式。
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            text = line[columns["text_a"]]
            text = [t for t in text.split()]
            src = args.tokenizer.convert_tokens_to_ids(text)    #转化为了input_ids
            seg = [0]
            i = 0
            while True:
                ip_seg = [i] * 11
                i += 1
                seg.extend(ip_seg)
                tcp_seg = [i] * 9
                i += 1
                seg.extend(tcp_seg)
                if len(seg) == len(src):
                    break
            if len(src) > args.seq_length:      #把超长向量截取成固定长度
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:   #把不够长的向量填充成固定长度
                src.append(0)
                seg.append(0)
            dataset.append((src, tgt, seg))     #token ID序列、标签、段落标识
    return dataset 

def get_input_ids(args):

    output_dir = \
    "~/llmfortraffic/IncrementalLearning/PyCIL/data/"  \
    +args['dataset_name']+"/"
    
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(f"{output_dir}train_src.npy"):
        return

    args = Namespace(**args)
    args.train_path = f"~/llmfortraffic/longtrafficformer" \
    f"/packetsformer/fine-data/fine_data_flow/{args.dataset_name}_enhance1/0.8/train_500_dataset.tsv"
    args.test_path = f"~/llmfortraffic/longtrafficformer" \
    f"/packetsformer/fine-data/fine_data_flow/{args.dataset_name}_enhance1/0.8/test_500_dataset.tsv"
    args.valid_path = f"~/llmfortraffic/longtrafficformer" \
    f"/packetsformer/fine-data/fine_data_flow/{args.dataset_name}_enhance1/0.8/valid_500_dataset.tsv"
    args.tokenizer = str2tokenizer['bert'](args)    #设置分词器
    args.seq_length = 401
    
    train_dataset = read_dataset(args, args.train_path)
    dev_dataset = read_dataset(args, args.valid_path)
    test_dataset = read_dataset(args, args.test_path)

    train_src = np.array([item[0] for item in train_dataset], dtype=np.int32)
    train_label = np.array([item[1] for item in train_dataset], dtype=np.int32)
    train_seg = np.array([item[2] for item in train_dataset], dtype=np.int32)
    
    test_src = np.array([item[0] for item in test_dataset], dtype=np.int32)
    test_label = np.array([item[1] for item in test_dataset], dtype=np.int32)
    test_seg = np.array([item[2] for item in test_dataset], dtype=np.int32)

    valid_src = np.array([item[0] for item in dev_dataset], dtype=np.int32)
    valid_label = np.array([item[1] for item in dev_dataset], dtype=np.int32)
    valid_seg = np.array([item[2] for item in dev_dataset], dtype=np.int32)
    # 保存为npy
    np.save(f"{output_dir}train_src.npy", train_src)      # (N, seq_length)
    np.save(f"{output_dir}train_label.npy", train_label)  # (N,)
    np.save(f"{output_dir}train_seg.npy", train_seg)      # (N, seq_length)
    
    np.save(f"{output_dir}test_src.npy", test_src)        # (N, seq_length)
    np.save(f"{output_dir}test_label.npy", test_label)    # (N,)
    np.save(f"{output_dir}test_seg.npy", test_seg)        # (N, seq_length)
    
    np.save(f"{output_dir}valid_src.npy", valid_src)      # (N, seq_length)
    np.save(f"{output_dir}valid_label.npy", valid_label)  # (N,)
    np.save(f"{output_dir}valid_seg.npy", valid_seg)      # (N, seq_length)

    args = vars(args)




def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())    # 获取模型所有命名参数(name, parameter)元组列表
    no_decay = ['bias', 'gamma', 'beta']    # 定义不需要权重衰减的参数名列表 权重衰减：在每次参数更新时，将权重乘以一个小于1的因子。 为了包含归一化的参数
    optimizer_grouped_parameters = [         # 将参数分为两组，用于不同的权重衰减策略
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]: # 根据指定的优化器类型创建优化器    默认为adamw     learning_rate默认为6e-5
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:       # 其他优化器：不使用参数缩放和相对步长
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:      # 根据指定的学习率调度策略创建调度器
        scheduler = str2scheduler[args.scheduler](optimizer)    # 常数学习率
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)  # 带预热的常数学习率
    else:   #scheduler默认为linear：LambdaLR
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)# 其他调度策略（如线性衰减）
    return optimizer, scheduler

def build_optimizer_scheduler(args, model):
    """构建UER-py的优化器和调度器"""
    # 重设训练步数（适配当前任务的样本数）
    args.train_steps = int(args.train_instances_num * 50 / args.batch_size) + 1
    optimizer, scheduler = build_optimizer(args, model)
    return [optimizer, scheduler]