#各种分类器头（SimpleLinear, CosineLinear等）
'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Optional, Dict
from abc import ABCMeta, abstractmethod


class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True, scale=10.0, use_cosine=False):
        super(SimpleLinear, self).__init__()
        self.use_cosine = use_cosine
        self.scale = scale  # 缩放因子，必须设置，否则数值太小
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias and not use_cosine: #余弦相似度通常不需要偏置（bias会破坏方向归一化）
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        if self.use_cosine:
            # 关键：L2归一化特征和权重
            input_norm = F.normalize(input, p=2, dim=1)
            weight_norm = F.normalize(self.weight, p=2, dim=1)
            
            # 计算余弦相似度，并缩放（否则数值在[-1,1]之间，太小）
            logits = torch.mm(input_norm, weight_norm.t()) * self.scale
            return {'logits': logits}
        else:
            return {'logits': F.linear(input, self.weight, self.bias)}


# 在文件末尾添加

class MLPClassifier(nn.Module):
    '''
    多层感知机分类头：增加表达能力，支持复杂数据分类
    结构: Linear → LayerNorm → ReLU → Dropout → Linear → Cosine
    '''
    def __init__(self, in_features, out_features, hidden_dim=None, dropout=0.1, use_cosine=False, scale=10.0):
        super(MLPClassifier, self).__init__()
        self.use_cosine = use_cosine
        self.scale = scale
        
        # 隐藏层维度：默认是输入维度的一半或至少128
        if hidden_dim is None:
            # hidden_dim = max(in_features // 2, 128)
            hidden_dim = in_features
        
        self.hidden_dim = hidden_dim
        
        # 第一层：特征变换
        self.fc1 = nn.Linear(in_features, hidden_dim)
        # self.norm1 = nn.LayerNorm(hidden_dim)
        # self.dropout1 = nn.Dropout(dropout)
        
        # 第二层：可选的另一个特征变换
        self.fc2 = nn.Linear(hidden_dim, out_features)
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.dropout2 = nn.Dropout(dropout)
        
        # 输出层
        # if use_cosine:      #默认为false
        #     self.weight = nn.Parameter(torch.Tensor(out_features, hidden_dim))
        #     self.register_parameter('bias', None)
        # else:
        #     self.weight = nn.Parameter(torch.Tensor(out_features, hidden_dim))
        #     self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        if hasattr(self, 'bias') and self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, input):
        # 第一层变换
        # x = self.fc1(input)
        # x = self.norm1(x)
        # x = F.relu(x)
        # x = self.dropout1(x)
        # # 第二层变换（可选）
        # x = self.fc2(x)
        # x = self.norm2(x)
        # x = F.relu(x)
        # x = self.dropout2(x)

        logits = self.fc2(torch.tanh(self.fc1(input)))
        
        # 输出层
        # if self.use_cosine:
        #     x_norm = F.normalize(x, p=2, dim=1)
        #     weight_norm = F.normalize(self.weight, p=2, dim=1)
        #     logits = torch.mm(x_norm, weight_norm.t()) * self.scale
        # else:
        #     logits = F.linear(x, self.weight, self.bias)
        
        return {'logits': logits}

class TagFex_SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TagFex_SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': reduce_proxies(out1['logits'], self.nb_proxy),
            'new_scores': reduce_proxies(out2['logits'], self.nb_proxy),
            'logits': out
        }


class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    """
    Abstract linear module for the analytic continual learning [1-3].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    [2] Zhuang, Huiping, et al.
        "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    [3] Zhuang, Huiping, et al.
        "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
    """

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super(torch.nn.Linear, self).__init__()  # Skip the Linear class
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma: float = gamma
        self.bias: bool = bias
        self.dtype = dtype

        # Linear Layer
        if bias:
            in_features += 1
        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        X = X.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)
        return {"logits": X @ self.weight}

    @property
    def in_features(self) -> int:
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        # Following the equation (4) of ACIL, self.weight is set to \hat{W}_{FCN}^{-1}
        self.weight = torch.zeros((self.weight.shape[0], 0)).to(self.weight)

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        raise NotImplementedError()

    def after_task(self) -> None:
        assert torch.isfinite(self.weight).all(), (
            "Pay attention to the numerical stability! "
            "A possible solution is to increase the value of gamma. "
            "Setting self.dtype=torch.double also helps."
        )


class RecursiveLinear(AnalyticLinear):
    """
    Recursive analytic linear (ridge regression) modules for the analytic continual learning [1-3].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    [2] Zhuang, Huiping, et al.
        "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    [3] Zhuang, Huiping, et al.
        "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
    """

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Regularized Feature Autocorrelation Matrix (RFAuM)
        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

    def update_fc(self, nb_classes: int) -> None:
        increment_size = nb_classes - self.out_features
        assert increment_size >= 0, "The number of classes should be increasing."
        tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
        self.weight = torch.cat((self.weight, tail), dim=1)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """The core code of the ACIL [1].
        This implementation, which is different but equivalent to the equations shown in the paper,
        which supports mini-batch learning.
        """
        X, Y = X.to(self.weight), Y.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)

        # ACIL
        # Please update your PyTorch & CUDA if the `cusolver error` occurs.
        # If you insist on using this version, doing the `torch.inverse` on CPUs might help.
        # >>> K_inv = torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T
        # >>> K = torch.inverse(K_inv.cpu()).to(self.weight.device)
        K = torch.inverse(torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T)
        # Equation (10) of ACIL
        self.R -= self.R @ X.T @ K @ X @ self.R
        # Equation (9) of ACIL
        self.weight += self.R @ X.T @ (Y - X @ self.weight)


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


'''
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return {'logits': out}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': out1['logits'],
            'new_scores': out2['logits'],
            'logits': out
        }
'''
