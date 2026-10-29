#BaseLearner基类，实现eval_task等通用方法
import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os
import sys

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1 #初始为-1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 1
        # self._grpoups = []

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.epoch_ = 0
        self.init_epoch_ = 0
        # self.args = args

    def recoder_epcho(self):    #slg
        if self.init_epoch_ != 0:
            logging.info("Init epoch: {}".format(self.init_epoch_))
        if self.epoch_ != 0:
            logging.info("Epoch: {}".format(self.epoch_))
    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class, adapters=False):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class, adapters)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        # self._grpoups.append(self._total_classes - self._known_classes)
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, 
                           10, self.args.get("increment_groups", None))
        
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]      #总体准确率
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        y_pred_top1 = y_pred.T[0]
        from sklearn.metrics import precision_score, recall_score, f1_score
        ret["precision"] = precision_score(y_true, y_pred_top1, average="macro")
        ret["recall"] = recall_score(y_true, y_pred_top1, average="macro")
        ret["f1"] = f1_score(y_true, y_pred_top1, average="macro")
        # logging.info('\n'+str(self._compute_confusion_matrix(y_pred_top1, y_true, 
        #                                                      max(y_true.max(),y_pred_top1.max())+1)))
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        print(str(self._compute_confusion_matrix(y_pred_top1, y_true, 
                                                max(y_true.max(),y_pred_top1.max())+1)))
        return ret

    def _evaluate_fc_simple(self, y_pred, y_true):
        ret = {}
        # self._grpoups.append(self._total_classes - self._known_classes)
        grouped = accuracy(y_pred, y_true, self._known_classes, 
                           10, self.args.get("increment_groups", None))
        
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]      #总体准确率
        y_pred_top1 = y_pred
        from sklearn.metrics import precision_score, recall_score, f1_score
        ret["precision"] = precision_score(y_true, y_pred_top1, average="macro")
        ret["recall"] = recall_score(y_true, y_pred_top1, average="macro")
        ret["f1"] = f1_score(y_true, y_pred_top1, average="macro")
        # logging.info('\n'+str(self._compute_confusion_matrix(y_pred_top1, y_true, 
        #                                                      max(y_true.max(),y_pred_top1.max())+1)))
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        print(str(self._compute_confusion_matrix(y_pred_top1, y_true, 
                                                max(y_true.max(),y_pred_top1.max())+1)))
        return ret

    def _compute_confusion_matrix(self, y_pred, y_true, num_classes):
        """
        y_pred: [N] 预测类别
        y_true: [N] 真实类别
        num_classes: 类别总数
        """
        cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[true_label, pred_label] += 1
        return cm.cpu().numpy()

    def eval_task(self, save_conf=False):
        # if self._cur_task == 5:
        #     import time
        #     loop_num = 10
        #     torch.cuda.synchronize()
        #     start_time = time.time()
        #     for i in range(loop_num):
        #         y_pred, y_true, y_fc, y_simple = self._eval_cnn(self.test_loader, test_speed=True)  # 执行原评估函数
        #     torch.cuda.synchronize()
        #     end_time = time.time()

        #     # ========== 3. 结果统计 ==========
        #     total_samples = len(y_true) * loop_num
        #     total_time = end_time - start_time
        #     throughput = total_samples / total_time  # 吞吐量：每秒处理样本数

        #     print("===== 推理速度统计 =====")
        #     print(f"总样本数: {total_samples}")
        #     print(f"端到端总耗时: {total_time:.2f} s")
        #     print(f"平均吞吐量: {throughput:.2f} 样本/秒")
        #     print(f"单样本平均耗时: {1000 / throughput:.2f} ms")
        y_pred, y_true, y_fc, y_simple = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        fc_accy = None
        simple_accy = None
        if len(y_fc) > 0:
            print("y_fc:")
            fc_accy = self._evaluate_fc_simple(y_fc, y_true)
            print("y_simple:")
            simple_accy = self._evaluate_fc_simple(y_simple, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy, fc_accy, simple_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                if "adapter_one" in self.args['model_name']:
                    outputs = model(inputs, _eval=True)["logits"]
                else:    
                    outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

            if os.environ.get("Pycil_Debug") == "1":
                break #slg debug

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader, test_speed=False):
        self._network.eval()
        y_pred, y_true = [], []
        y_fc_pred = []
        y_simple_pred = []
        if "simple" in self.args and self.args["simple"]:
            simple = True
        else:
            simple = False
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                if "adapter_one" in self.args['model_name']:
                    outputs_all = self._network(inputs, _eval=True, simple=simple)
                    outputs = outputs_all["logits"]
                else:
                    outputs = self._network(inputs,)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            if simple:  #如果启用simple分类头
                torch.set_printoptions(
                    threshold=1000000,    # 足够大
                    edgeitems=1000000,
                    linewidth=1000000,    # 关键！别用 inf，用 100万 就行
                    precision=4,
                    sci_mode=False
                )
                y_simple = []
                y_fc = []
                for i in range(len(outputs_all['log_fc'])):
                    log_fc = outputs_all['log_fc'][i].detach().cpu().numpy()
                    log_simple = outputs_all['log_simple'][i].detach().cpu().numpy()
                    y_simple.append(int(log_simple.argmax()))
                    y_fc.append(int(log_fc.argmax()))
                    if int(targets[i].item()) == int(predicts[i].item()) \
                    == int(log_fc.argmax()) == int(log_simple.argmax()) or test_speed:
                        continue
                    print("y_true:", targets[i].item(), "\t\ty_pred:", predicts[i].item(), "\t\tfc_index:",log_fc.argmax()
                          ,"\t\tsimple_index:", log_simple.argmax())
                    # positions = list(range(len(log_fc)))
                    print("pos:    ", end="")
                    for idx in range(len(log_fc)):
                        print(f"{idx:>7}", end="")  # 索引右对齐，占7个字符
                    print()
                    
                    print("log_fc:  ", end="")
                    for v in log_fc:
                        print(f"{v:>7.4f}", end="")  # 数值右对齐，占7个字符，4位小数
                    print()
                    
                    print("log_smp: ", end="")
                    for v in log_simple:
                        print(f"{v:>7.4f}", end="")
                    print()
                    print("-" * 140)
                
                y_fc_pred.append(y_fc)
                y_simple_pred.append(y_simple)


            if os.environ.get("Pycil_Debug") == "1" and not test_speed:
                break #slg debug
        if simple:
            return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(y_fc_pred), np.concatenate(y_simple_pred)  # [N, topk]
        else:
            return np.concatenate(y_pred), np.concatenate(y_true), [], []  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):     #提取样本特征向量 的辅助函数。
        self._network.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

                if os.environ.get("Pycil_Debug") == "1":
                    break #slg  debug

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        # logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m): #保留旧样本
        # logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m, adapters=False): 
        """
        统一构建新旧类别的原型样本(Exemplar)，并计算所有类别的特征均值
        Args:
            data_manager: 数据集管理工具，负责加载/处理数据
            m:  每个类别保留的原型样本数量
            adapters: 是否有多个adapter进行并行推理
        """
        if not adapters:
            # 1. 初始化类别均值矩阵：形状 [总类别数, 特征维度]，全0         10， 768
            _class_means = np.zeros((self._total_classes, self.feature_dim))
            
            # ===================== 第一部分：处理【旧已知类别】，重新计算特征均值 =====================
            # 遍历所有已经学习过的旧类别
            for class_idx in range(self._known_classes):
                mask = np.where(self._targets_memory == class_idx)[0]   # 从记忆库中筛选出当前旧类别的所有样本索引

                class_data, class_targets = (   # 从记忆库中取出当前类别的 样本数据 和 标签
                    self._data_memory[mask],
                    self._targets_memory[mask],
                )

                class_dset = data_manager.get_dataset(  # 用数据集管理器构建测试模式的数据集（不做数据增强）
                    [], source="train", mode="test", appendent=(class_data, class_targets)
                )
                class_loader = DataLoader(  # 构建数据加载器：批处理、不打乱、4个线程加载
                    class_dset, batch_size=batch_size, shuffle=False, num_workers=4
                )

                # 2. 用当前训练好的网络提取样本的特征向量（核心：网络更新后重新提取旧类特征）
                vectors, _ = self._extract_vectors(class_loader)    
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T # 对特征做【L2归一化】：每个样本的特征向量长度归一化为1 (防止特征尺度影响)
                mean = np.mean(vectors, axis=0)     # 计算当前类别所有样本特征的均值
                mean = mean / np.linalg.norm(mean)      # 对类别均值再做一次L2归一化

                _class_means[class_idx, :] = mean   # 将归一化后的类别均值存入矩阵对应位置

                if os.environ.get("Pycil_Debug") == "1":
                    break #slg  debug


        # ===================== 第二部分：处理【新类别】，选择原型样本 + 计算均值 =====================
        # 遍历所有新类别（从已知类后一个 到 总类别）
        for class_idx in range(self._known_classes, self._total_classes):

            # 加载当前新类别的【全部训练数据】、标签、数据集
            data, targets, class_dset = data_manager.get_dataset(   
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(  # 构建新类别的数据加载器
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # 提取新类别所有样本的特征向量
            vectors, _ = self._extract_vectors(class_loader)
            # 特征L2归一化
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # 计算该类别【全部样本】的真实特征均值
            class_mean = np.mean(vectors, axis=0)

            # ---------------- 核心：贪心算法选择 m 个原型样本 ----------------
            selected_exemplars = []     # 存储选中的原型样本（原始数据）
            exemplar_vectors = []       # 存储选中样本的特征向量

            # 循环选择 m 个样本（k=1到m）
            m = min(m, len(data))  # 如果样本总数不足 m 个，则选择全部样本
            for k in range(1, m + 1):
                S = np.sum( # 计算已选中样本的特征向量之和
                    exemplar_vectors, axis=0
                ) 

                # 候选均值：(剩余所有样本特征 + 已选样本和) / 当前选中数量k     每一行代表：如果我选这个候选样本，当前 k 个样本的均值是多少
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors

                # 找到：让候选均值最接近【类别真实均值】的样本索引i         np.argmin返回最小值索引
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(  # 将选中的样本存入结果（深拷贝，避免引用传递）
                    np.array(data[i])
                )  
                exemplar_vectors.append(    # 将选中样本的特征存入结果
                    np.array(vectors[i])
                )  

                # 从候选池删除已选中的样本，避免重复选择
                vectors = np.delete(    
                    vectors, i, axis=0
                ) 
                data = np.delete(
                    data, i, axis=0
                ) 

                # if os.environ.get("Pycil_Debug") == "1":
                #     break #slg  debug
            # ---------------- 将选中的原型样本存入记忆库 ----------------
            
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)    # 为原型样本生成标签（全为当前新类别）
            self._data_memory = (   # 拼接：将新原型样本添加到全局记忆库（旧样本+新样本）
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (    # 拼接标签到记忆库
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
            if not adapters:
                # Exemplar mean
                # ---------------- 计算新类别【原型样本】的均值 ----------------
                exemplar_dset = data_manager.get_dataset(   # 构建仅包含原型样本的数据集
                    [],
                    source="train",
                    mode="test",
                    appendent=(selected_exemplars, exemplar_targets),
                )
                exemplar_loader = DataLoader(
                    exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
                )
                vectors, _ = self._extract_vectors(exemplar_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

                _class_means[class_idx, :] = mean

            if os.environ.get("Pycil_Debug") == "1":
                break #slg  debug

        # 3. 最终保存所有类别的归一化均值（用于后续分类）
        if not adapters:
            self._class_means = _class_means
