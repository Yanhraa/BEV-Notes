# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    # src:CNN模型输出(256)
    # mask
    def forward(self, src, mask, query_embed, pos_embed):
        # TODO: 实现 Transformer 模型的前向传播逻辑
        # 1. 将输入展平，将形状从 (bs, c, h, w) 变为 (hw, bs, c)

        # bs:batch_size
        # c:channels
        bs, c, h, w = src.shape

        # flatten().permute(, , )用于张量展平和重新排列张量维度
        # flatten(2).permute(2, 0, 1):(bs, c, h, w)->(h*w, bs, c)
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        # 2. 初始化需要预测的目标 query embedding

        # unsqueeze(1).repeat(1, bs, 1) 在指定位置增加一个维度，并在指定的维度重复张量的元素
        # (1, bs, 1) 1：不重复 ,bs：重复bs次
        # (L, D)->(L, bs, D) L 查询嵌入的长度 D 嵌入的维度
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # (bs, h, w)->(bs, h*w)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        # 3. 使用编码器处理输入序列，得到具有全局相关性（增强后）的特征表示
        '''     
            encoder
            def forward(self, src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None):
        '''
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # 4. 使用解码器处理目标张量和编码器的输出，得到output embedding

        '''
            def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        '''
        output = self.decoder(tgt, memory, memory_key_padding_mask=mask, 
                          pos=pos_embed, 
                          query_pos=query_embed)

        # 5. 对输出结果进行形状变换，并返回
        # decoder输出 [1, 100, bs, 256] -> [1, bs, 100, 256]
        # encoder输出 [bs, 256, H, W]

        # ()
        return output.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w) 
        # pass


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # self.layers包含num_layers个encoder_layer副本
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # TODO: 实现 Transformer 编码器的前向传播逻辑
        # (h*w, bs, c)
        output = src
        # 1. 遍历$num_layers层TransformerEncoderLayer
        # 迭代编码器层 layer 是 TransformerEncoder对象实例
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        # 2. layer norm（如果self.norm不为None）
        if self.norm is not None:
        # 对输出层进行归一化
            output = self.norm(output)
        # 3. 得到最终编码器的输出
        return output
        # pass


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # 是否返回中间层输出结果
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # TODO: 实现 Transformer 解码器的前向传播逻辑
        # 1. 遍历$num_layers层TransformerDecoderLayer，对每一层解码器进行前向传播，
        # 并处理return_intermediate为True的情况
        output = tgt
        # 用于存储中间层的输出
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # 2. 应用最终的归一化层layer norm
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
            
        # 3. 如果设置了返回中间结果，则将它们堆叠起来返回；否则，返回最终输出
        if self.return_intermediate:
            # 通过stack堆叠成一个张量并返回
            return torch.stack(intermediate)
            
        return output.unsqueeze(0)
        # pass


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        src: [494, bs, 256]  backbone输入下采样32倍后 再压缩维度到256的特征图
        src_mask: None，在Transformer中用来“防作弊”,即遮住当前预测位置之后的位置，忽略这些位置，不计算与其相关的注意力权重
        src_key_padding_mask: [bs, 494]  记录backbone生成的特征图中哪些是原始图像pad的部分 这部分是没有意义的
                           计算注意力会被填充为-inf，这样最终生成注意力经过softmax时输出就趋向于0，相当于忽略不计
        pos: [494, bs, 256]  位置编码
        """
        # TODO: 实现 Transformer 编码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）
        q = k = self.with_pos_embed(src, pos)
        src_msa = self.self_attn(query=q, key=k, value=src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)[0]
        output = src + self.dropout1(src_msa)
        output = self.norm1(output)
        src_msa = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout2(src_msa)
        output = self.norm2(output)

        return output
        # pass 
 

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """
        src: [494, bs, 256]  backbone输入下采样32倍后 再压缩维度到256的特征图
        src_mask: None，在Transformer中用来“防作弊”,即遮住当前预测位置之后的位置，忽略这些位置，不计算与其相关的注意力权重
        src_key_padding_mask: [bs, 494]  记录backbone生成的特征图中哪些是原始图像pad的部分 这部分是没有意义的
                           计算注意力会被填充为-inf，这样最终生成注意力经过softmax时输出就趋向于0，相当于忽略不计
        pos: [494, bs, 256]  位置编码
        """
        # TODO: 实现 Transformer 编码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）
        # 而是在此处对输入进行LN 
        src_msa = src.norm1(src)
        # 加上位置编码
        q = k = self.with_pos_embed(src_msa, pos)
        # MSA:self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 也许此处不需要attn_mask，因为做的就是全局的注意力
        # class
        src_msa = self.self_attn(query=q, key=k, value=src_msa, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)[0]
        # Self Attention与残差连接，并应用dropout防止过拟合
        output = src + self.dropout1(src_msa)
        # 对残差连接结果进行LN
        output = self.nor2(output)
        # 进行线性变换，激活函数处理，然后应用dropout
        src_msa = self.linear2(self.dropout(self.activation(self.linear1(output))))
        # 将线性变换后的输出与残差连接的结果相加
        output = output + self.dropout2(src_msa)
        return output
        # pass

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            # 先对输入进行LN 
            # ViT结构中不应该在此处进行LN
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        tgt: 需要预测的目标 query embedding，负责预测物体
        memory: [h*w, bs, 256]， Encoder的输出，具有全局相关性（增强后）的特征表示
        tgt_mask: None
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]  encoder输出特征图的位置编码
        query_pos: [100, bs, 256]  query embedding/tgt的位置编码  负责建模物体与物体之间的位置关系  随机初始化的
        tgt_mask、memory_mask、tgt_key_padding_mask是防止作弊的 这里都没有使用
        """
        # TODO: 实现 Transformer 解码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt_mha = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)[0]
        output = tgt+self.dropout1(tgt_mha)
        output = self.norm1(output)
        # memory: [h*w, bs, 256]， Encoder的输出，具有全局相关性（增强后）的特征表示
        k = self.with_pos_embed(memory, pos)
        q = self.with_pos_embed(output, query_pos)
        tgt_mha = self.multihead_attn(query=q,
                                      key=k,
                                      value=memory, attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)[0]
        
        output = output + self.dropout2(tgt_mha)
        output = self.norm2(output)
        tgt_mha = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout3(tgt_mha)
        output = self.norm3(output)


        return output
        # pass

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
        tgt: 需要预测的目标 query embedding，负责预测物体
        memory: [h*w, bs, 256]， Encoder的输出，具有全局相关性（增强后）的特征表示
        tgt_mask: None
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]  encoder输出特征图的位置编码
        query_pos: [100, bs, 256]  query embedding/tgt的位置编码  负责建模物体与物体之间的位置关系  随机初始化的
        tgt_mask、memory_mask、tgt_key_padding_mask是防止作弊的 这里都没有使用
        """
        # TODO: 实现 Transformer 解码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）
        # 此处tgt应该是zeros
        # tgt = torch.zeros_like(query_embed)
        # 还有必要做ln吗？
        tgt_mha = self.norm1(tgt)

        q = k = self.with_pos_embed(tgt_mha, query_pos)
        tgt_mha = self.self_attn(query=q, key=k, value=tgt_mha,
                                 key_padding_mask=tgt_key_padding_mask)[0]
        output = tgt+self.dropout1(tgt_mha)
        output = self.norm2(output)
        # memory: [h*w, bs, 256]， Encoder的输出，具有全局相关性（增强后）的特征表示
        tgt_mha = self.multihead_attn(query=self.with_pos_embed(output, query_pos),
                                      key=self.with_pos_embed(memory, pos),
                                      value=memory, attn_msk=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)[0]
        
        output = output + self.dropout2(tgt_mha)
        tgt_mha = self.norm3(output)
        tgt_mha = self.linear2(self.dropout(self.activation(self.linear1(tgt_mha))))
        output = output + self.dropout3(tgt_mha)



        return output
        # pass

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            # 先对输入进行LayerNorm
            # tgt = tgt.norm1(tgt)
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

# 复制对象n次
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
