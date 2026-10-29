'''
transformer 最后多层加 adapter 并冻结主体，
每次增量任务增加多个 adapter 和对应的分类头。
推理时并行推理，取最大的 logits 作为结果。


保留样本数减少、kl_loss系数减小
'''

# import logging
import copy
import numpy as np
# from tqdm import tqdm
import torch
# from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy
from itertools import cycle
import os

EPSILON = 1e-8
if os.environ.get("Pycil_Debug") == "0":      #没有debug
    epochs = 50      #200      50
    epochs_old = 30
else:
    epochs = 1
    epochs_old = 1
batch_size = 32
lrate = 1e-4  #0.001   6e-5
weight_decay = 0.01 #5e-4

lrate_old = 5e-5  #0.001   6e-5
weight_decay_old = 0.001 #5e-4
num_workers = 0
step_size = 20      #45
gamma = 0.5         #0.1
class ADAPTER_ONE_FEATURE(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False,adapter_one=True, fc_balance=True)
        self.old_known_class = []       #实际类别数
        self.old_fc_out = [0]   #实际分类数

    def after_task(self):
        self._known_classes = self._total_classes
        self.init_epoch_ = epochs
        self.epoch_ = epochs_old

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1         ##初始为-1
        self.old_known_class.append(self._total_classes)    #_total_classes初始为0

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        

        self._network_expansion()   #添加adapter和分类头 更新分类头 冻结了偏置层

        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # valid_dataset = data_manager.get_dataset(
        #     np.arange(0, self._total_classes), source='valid', mode='valid')
        # self.valid_loader = DataLoader(
        #     valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(
            self.train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            
        self._train(self.train_loader, epochs=epochs)

        self.build_rehearsal_memory(data_manager, self.samples_per_class, adapters=True)   #保存旧类样本
        
        #依次训练前面的几个分类头，重新加载train_loader
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                            mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        for order in range(len(self._network.fc_num)-1):        #前面的分类头进行遍历更新
            print(f"old_adapter_order: {order}")
            self._old_network_expansion(order)
            self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self.old_known_class[order],
                                                                            self.old_known_class[order+1]),source="train", mode="test")
            self.train_loader_for_protonet = DataLoader(
                self.train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self._train(self.train_loader, order, epochs_old)

    def _old_network_expansion(self, order):
        '''
        解冻前面的adapter和fc层
        order:第几个任务
        '''
        for p in self._network.convnet.parameters():    #全部进行冻结
            p.requires_grad = False
        if len(self._network.fc_group) > 0:     #冻结所有的fc_group
            for p in self._network.fc_group.parameters():
                p.requires_grad = False
        current_adapters = self._network.convnet.adapter_groups[order]
        for p in current_adapters.parameters():
            p.requires_grad = True

        fc_layer = self._network.fc_group[order]    #解冻第fc_order个任务的fc_group
        for p in fc_layer.parameters():
            p.requires_grad = True
        self._network.convnet.switch("adapter_one")  

        
    def _train(self, train_loader, order=-1, epochs=20):
        '''
        order:要训练的分类头索引
        '''
        self._network.to(self._device)
        self._epoch_num = epochs

        base_params = [p for name, p in self._network.named_parameters() if 'adapter' in name and p.requires_grad]
        base_fc_params = [p for name, p in self._network.named_parameters() if 'adapter' not in name and p.requires_grad]
        base_params = {'params': base_params, 'lr': 0.03, 'weight_decay': 0.0005}
        base_fc_params = {'params': base_fc_params, 'lr': 0.03 *0.1, 'weight_decay': 0.0005}
        network_params = [base_params, base_fc_params]
        optimizer = torch.optim.SGD(
            network_params, 
            momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self._epoch_num, eta_min=1e-8)
        self._train_function(train_loader, optimizer, scheduler, order)

        print("\n\n")
        self.replace_fc(order)   #使用protonet的方式替换fc层的权重

        # torch.save(self._network.state_dict(), f"logs/adapter_one/{self.args['dataset_name']}/{self._cur_task}_{self.args['pretrained']}_2.dict")

    def replace_fc(self, order=-1):       
        model = self._network.to(self._device)
        self._network.eval()    #replace_fc 时网络在 train 模式，dropout 干扰了特征分布。
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader_for_protonet):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.forward_orig(data, order)       #就是为了得到模块提取的feature
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
                if os.environ.get("Pycil_Debug") == "1":
                    break #slg  debug
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        class_list = np.unique(self.train_dataset_for_protonet.labels)
        for class_index in class_list:
            if class_index <  self.old_known_class[order]:      #老类的simplecil不进行更新
                continue
            if order != -1 and class_index >= self.old_known_class[order+1]:
                continue
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc_simple_group[order].weight.data[class_index-self.old_known_class[order]] = proto
        return model

    def _train_function(self, train_loader, optimizer, scheduler, order=-1):
        for _, epoch in enumerate(range(self._epoch_num)):
            self._network.train()
            losses, loss_clfs  = 0., 0.
            kl_losses = 0.
            loss_orthes = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                logits_new, targets_new, loss_clf, kl_loss, loss_orth = self._compute_ssre_loss(inputs,targets, order)#总损失 = 分类损失
                loss = loss_clf + kl_loss + loss_orth
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                loss_clfs += loss_clf.item()
                kl_losses += kl_loss.item()
                loss_orthes += loss_orth.item()

                if logits_new is not None:
                    _, preds = torch.max(logits_new, dim=1)
                    # correct += preds.eq(targets_new.expand_as(preds)).cpu().sum()
                    correct += preds.eq(targets_new.expand_as(preds)).sum().item()
                    total += len(targets_new)
                if os.environ.get("Pycil_Debug") == "1":
                    break #slg  debug
            

            scheduler.step()
            # train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if total == 0:
                total = 1
            train_acc = np.around(correct*100 / total, decimals=2)
            train_loader_len = len(train_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.6f}, Loss_clf  {:.6f}, kl_loss {:.6f},' \
            ' loss_orth {:.6f},Train_accy {:.2f}'.format(
                        self._cur_task, epoch+1, self._epoch_num, 
                        losses/train_loader_len, loss_clfs/train_loader_len, 
                        kl_losses/train_loader_len, loss_orthes/train_loader_len,train_acc)
            print(info)

    def _compute_ssre_loss(self,inputs, targets, order=-1):
        '''
        order:要训练的分类头索引
        '''
        outputs = self._network(inputs)
        logits = outputs["logits"]
        # features = outputs["features"][-1]
        features = outputs["features"][order]

        logits_new = None
        targets_new = None
        #局部loss
        if order == -1:
            logits = logits[:,-self._network.fc_out[-1]:]   #选取最后一个fc的分类logits
            targets = targets - self._known_classes
        else:
            logits = logits[:,self.old_fc_out[order]:self.old_fc_out[order+1]]
            targets = targets - self.old_known_class[order]
        #loss_clf  
        valid_mask = (targets >= 0) & (targets < self._network.fc_num[order])   #超出分类范围的都为负类
        invalid_mask = ~valid_mask  # 旧类样本掩码
        num_new_classes = logits.shape[1]       #self._network.fc_num[-1]
        # 1. 新类样本：正常交叉熵
        loss_clf = torch.tensor(0.0, device=self._device, requires_grad=True)
        loss_uniform = torch.tensor(0.0, device=self._device, requires_grad=True)
        if valid_mask.sum() > 0:    #有新类样本 一个新类样本都没有，全是旧类就跳过
            loss_clf = F.cross_entropy(logits[valid_mask], targets[valid_mask])
            logits_new = logits[valid_mask]
            targets_new = targets[valid_mask]

        # 2. 旧类样本：强制均匀输出（KL散度损失）
        if invalid_mask.sum() > 0:
            uniform_target = torch.ones_like(logits[invalid_mask]) / num_new_classes
            # KL散度：让旧类样本的logits逼近均匀分布
            loss_uniform = F.kl_div(F.log_softmax(logits[invalid_mask], dim=1), uniform_target, reduction="batchmean")

        loss_orth = torch.tensor(0.0, device=self._device, requires_grad=True)

        # 获取当前批次所有特征和标签（包含旧类+新类）
        all_feats = features
        all_labels = targets

        # 仅当批次中 ≥2个不同类别时，计算正交损失（单个类别无法正交）
        unique_labels = torch.unique(all_labels)
        if len(unique_labels) >= 2:
            # 2. 计算【每个标签的特征中心】（同类样本特征均值）
            # 包含：旧类中心(target<0) + 新类中心(target≥0)
            class_centers = []
            for lab in unique_labels:
                # 筛选当前类别的所有特征
                mask_lab = (all_labels == lab)
                lab_feats = all_feats[mask_lab]
                # 计算类别中心
                center = lab_feats.mean(dim=0, keepdim=True)
                class_centers.append(center)        #各个类别样本的均值
            class_centers = torch.cat(class_centers, dim=0)  # [num_classes, feat_dim]
            centers_norm = F.normalize(class_centers, p=2, dim=1)
            cos_sim = torch.mm(centers_norm, centers_norm.t())
            num_classes = cos_sim.shape[0]
            mask = ~torch.eye(num_classes, dtype=torch.bool, device=self._device)
            '''
            mask生成矩阵
            [[False, True, True],
            [True, False, True],
            [True, True, False]]
            '''
            inter_cos_sim = cos_sim[mask]  # 取出所有不同类之间的余弦相似度 只取出非对角线的值
            # 4. 损失函数：最小化不同类的余弦相似度 → 让类别尽可能分开
            loss_orth = inter_cos_sim.pow(2).mean()  # 均方误差，最小化非对角元素平方和
        return logits_new, targets_new, loss_clf, loss_uniform, loss_orth
        #                                交叉熵     均匀损失      正交损失                    
    
    def _network_expansion(self):
        for p in self._network.convnet.parameters():    #全部进行冻结
            p.requires_grad = False
        if len(self._network.fc_group) > 0:     #冻结之前任务的fc_group
            for p in self._network.fc_group.parameters():
                p.requires_grad = False

        self._network.convnet._add_new_adapter_layers(adapter_layer_num=self.args["adapter_num"])  #每个任务对应多个个adapter 解冻新添加的 Adapter
        current_adapters = self._network.convnet.adapter_groups[-1]
        for p in current_adapters.parameters():
            p.requires_grad = True

        add_cls = self._total_classes - self._known_classes   #平衡每个类别的分类数
        if  add_cls < self.args['init_cls']:
            new_cls = self.args['init_cls']
        else:
            new_cls = add_cls
        self.old_fc_out.append(self.old_fc_out[-1] + new_cls)   #更新实际分类数
        self._network.update_fc_group(new_cls, add_cls, use_mlp=True, simple=True)

        currernt_fc = self._network.fc_group[-1]    #每个任务对应一个分类头  解冻新添加的分类头
        for p in currernt_fc.parameters():       
            p.requires_grad = True

        #强制清理CUDA缓存（在关键节点使用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放PyTorch缓存的未使用显存

        if len(self._network.fc_simple_group) > 0:
            for p in self._network.fc_simple_group.parameters():
                p.requires_grad = False
        self._network.convnet.switch("adapter_one")     
        