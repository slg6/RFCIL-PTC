import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        
        
        if args.target in ['bert','bertflow','mlm'] and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight
        elif args.target in ['lm','t5'] and args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight

        if args.target == 't5' and args.share_embedding:
            self.target.embedding.word_embedding.weight = self.embedding.word_embedding.weight
        
        self.is_moe = args.is_moe

    def forward(self, src, tgt, seg, proto=None):
        emb = self.embedding(src, seg)
        if self.is_moe:
            output,gate_loss = self.encoder(emb, seg, src, proto)
            loss_info = self.target(output, tgt) + (gate_loss,)
        else:
            output = self.encoder(emb, seg) #这一步会让显存占用暴增
            loss_info = self.target(output, tgt)
        return loss_info

def from_pretrained(self,args, model):
    try:
        # 尝试加载预训练模型的权重
        pretrained_weights = torch.load(args.pretrained_model_path)
        model_dict = self.state_dict() #你可以直接修改 state_dict 中的张量，然后使用 load_state_dict() 将修改后的权重加载到模型中。
        # 过滤掉不匹配的权重
        filtered_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 更新模型的权重
        model_dict.update(filtered_weights)
        self.load_state_dict(model_dict)
        print("Successfully loaded pre-trained model weights.")
    except Exception as e:
        print(f"Failed to load pre-trained model weights: {e}")
        print("Initializing model parameters randomly.")
        # 如果加载失败，可以随机初始化模型参数
        self.apply(model._init_weights)