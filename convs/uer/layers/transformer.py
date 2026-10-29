import torch.nn as nn
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward, GatedFeedForward, FeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.relative_position_embedding import RelativePositionEmbedding
from uer.layers.moe_layer import MoELayer

class TransformerLayer(nn.Module):  #这是 Transformer 编码器中的一个基本构建块，多个这样的层堆叠形成完整的编码器。
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num    # 多头注意力机制的头数

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(      #这是多头注意力机制(Multi-Head Attention)层
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Feed forward layer.
        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(        #这是前馈神经网络(Feed Forward Network, FFN)层
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        self.dropout_2 = nn.Dropout(args.dropout)

        if args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)   #T5LayerNorm是Transformer中的LayerNorm
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)     # 第一个LayerNorm：用于注意力子层。 LayerNorm是归一化层
            self.layer_norm_2 = LayerNorm(args.hidden_size)     # 第二个LayerNorm：用于前馈网络子层



    def forward(self, hidden, mask, position_bias = None, use_adapters=False,
                moe_adapters=False, current_task_id=None, adapter_layer=None,
                adapter_layer_group=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
            use_adapters: 是否使用 Adapter
            use_all_adapters: 是否使用所有 Adapter（多专家累积）
            current_task_id: 当前任务 ID（仅使用当前任务 Adapter）
            adapter_layer:只在某一层的transformer中添加adapter模块
            adapter_layer_group:在多层transformer中添加adapter模块
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        # if self.layernorm_positioning == "post":    #default:post
        if adapter_layer_group is not None:   #所有的adapter都进行推理,在多层transformer中添加
            outputs = []
            if isinstance(hidden, list):
                for i in range(len(self.adapters)):
                    hidden_ = hidden[i]
                    adapter = self.adapters[i]
                    inter,probs = self.self_attn(hidden_, hidden_, hidden_, mask, position_bias)
                    inter = self.dropout_1(inter)
                    inter = self.layer_norm_1(inter + hidden_)
                    ffn_out = self.feed_forward(inter)      #ffn
                    adapter_residual = adapter(inter)
                    output = self.dropout_2(ffn_out + adapter_residual)
                    output = self.layer_norm_2(output + inter)  #laynormal
                    outputs.append(output)
            else:
                inter,probs = self.self_attn(hidden, hidden, hidden, mask, position_bias)
                inter = self.dropout_1(inter)
                inter = self.layer_norm_1(inter + hidden)
                ffn_out = self.feed_forward(inter)      #ffn
                for i, adapter in enumerate(self.adapters):
                    adapter_residual = adapter(inter)
                    output = self.dropout_2(ffn_out + adapter_residual)
                    output = self.layer_norm_2(output + inter)  #laynormal
                    outputs.append(output)
            return outputs,probs

                            
        inter,probs = self.self_attn(hidden, hidden, hidden, mask, position_bias)
        inter = self.dropout_1(inter)
        inter = self.layer_norm_1(inter + hidden)
        ffn_out = self.feed_forward(inter)      #ffn

        #adapter
        if use_adapters and hasattr(self, 'adapters') and len(self.adapters) > 0:
            # 累加所有 Adapter 的残差（知识累积）
            adapter_residual = 0
            for i, adapter in enumerate(self.adapters):
                if adapter is not None:
                    adapter_residual = adapter_residual + adapter(inter)
            ffn_out = ffn_out + adapter_residual
        if moe_adapters and current_task_id is not None:    #专家模式，只需要启用对应任务的adapter即可,task 0没有创建adapter
            adapter_residual = 0    #current_task_id为None时时ta_net 提取通用特征，不需要adapter参与
            adapter_residual = self.adapters[current_task_id](inter)
            ffn_out = ffn_out + adapter_residual

        if adapter_layer is not None:   #所有的adapter都进行推理,只在最后一层添加
            adapter_residual = 0
            outputs = []
            for i, adapter in enumerate(self.adapters):
                adapter_residual = adapter(inter)
                # ffn_out = ffn_out + adapter_residual
                output = self.dropout_2(ffn_out + adapter_residual)
                output = self.layer_norm_2(output + inter)  #laynormal
                outputs.append(output)
            return outputs,probs


            # if use_all_adapters:
            #     # 推理时：累加所有专家 Adapter
            #     for adapter in self.adapters:
            #         if adapter is not None:
            #             adapter_residual = adapter_residual + adapter(inter)
            # elif current_task_id is not None and current_task_id < len(self.adapters):
            #     # 训练时：只使用当前任务的 Adapter
            #     adapter_residual = self.adapters[current_task_id](inter)
        #adapter

        output = self.dropout_2(ffn_out)
        output = self.layer_norm_2(output + inter)  #laynormal
        # else:
        #     inter = self.layer_norm_1(hidden)
        #     inter,probs= self.self_attn(inter, inter, inter, mask, position_bias)
        #     inter = self.dropout_1(inter)
        #     hidden = hidden + inter
        #     output = self.layer_norm_2(hidden)
        #     output = self.dropout_2(self.feed_forward(output)) + hidden
        return output,probs

class TransformerMOELayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args):
        super(TransformerMOELayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Feed forward layer.
        ffn = FeedForward(args.hidden_size, args.moebert_expert_dim, args.hidden_act, args.dropout)
        self.experts = MoELayer(
                hidden_size=args.hidden_size,
                expert=ffn,
                num_experts=args.moebert_expert_num,
                route_method=args.moebert_route_method,
                vocab_size=args.vocab_size,
                hash_list=args.moebert_route_hash_list,
            )
        if args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)



    def forward(self, hidden, mask, position_bias = None, expert_input_ids=None, proto=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask, position_bias))
        inter = self.layer_norm_1(inter + hidden)
        output, balance_loss, gate_load = self.experts(inter,expert_input_ids,mask,proto=proto)

        return output, balance_loss

class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Multi-headed context-attention.
        self.context_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        # Feed forward layer.
        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        self.dropout_3 = nn.Dropout(args.dropout)

        # Layer Normalization
        if  args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(args.hidden_size)
            self.layer_norm_2 = T5LayerNorm(args.hidden_size)
            self.layer_norm_3 = T5LayerNorm(args.hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(args.hidden_size)
            self.layer_norm_2 = LayerNorm(args.hidden_size)
            self.layer_norm_3 = LayerNorm(args.hidden_size)



    def forward(self, hidden, encoder_hidden, mask_decoder, mask_encoder, self_position_bias = None, context_position_bias = None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            encoder_hidden: [batch_size x seq_length x emb_size]
            mask_encoder: [batch_size x 1 x seq_length x seq_length]
            mask_decoder: [batch_size x 1 x seq_length x seq_length]
            self_position_bias: [1 x heads_num x seq_length x seq_length]
            context_position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.layernorm_positioning == "post":
            query = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask_decoder, self_position_bias))
            query_norm = self.layer_norm_1(query + hidden)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias))
            mid_norm = self.layer_norm_2(mid + query_norm)
            output = self.dropout_3(self.feed_forward(mid_norm))
            output = self.layer_norm_3(output + mid_norm)
        else:
            hidden_norm = self.layer_norm_1(hidden)
            query = self.dropout_1(self.self_attn(hidden_norm, hidden_norm, hidden_norm, mask_decoder, self_position_bias))
            query = query + hidden
            query_norm = self.layer_norm_2(query)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, query_norm, mask_encoder, context_position_bias))
            mid = mid + query
            mid_norm = self.layer_norm_3(mid)
            output = self.dropout_3(self.feed_forward(mid_norm)) + mid
        return output


#class GptBlock(nn.Module):
#    def __init__(self, args):
#        super(GptBlock, self).__init__()

#        # Multi-headed self-attention.
#        self.self_attn = MultiHeadedAttention(
#            args.hidden_size, args.heads_num, args.dropout
#        )
#        self.layer_norm_1 = LayerNorm(args.hidden_size)
#        # Feed forward layer.
#        self.feed_forward = PositionwiseFeedForward(
#            args.hidden_size, args.feedforward_size, args.hidden_act
#        )
#        self.layer_norm_2 = LayerNorm(args.hidden_size)

#    def forward(self, hidden, mask):
#        """
#        Args:
#            hidden: [batch_size x seq_length x emb_size]
#            mask: [batch_size x 1 x seq_length x seq_length]
#        Returns:
#            output: [batch_size x seq_length x hidden_size]
#        """
#        inter = self.layer_norm_1(hidden)
#        inter = self.self_attn(inter, inter, inter, mask)
#        hidden = hidden + inter
#        output = self.layer_norm_2(hidden)
#        output = self.feed_forward(output)
        
#        return output + hidden
