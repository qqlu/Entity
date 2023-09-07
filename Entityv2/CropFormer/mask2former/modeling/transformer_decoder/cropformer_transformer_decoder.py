# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine3D2D
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

import pdb

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Make3dQueries(nn.Module):
    _version = 2
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enc_crosattn_3d = nn.ModuleList()
        self.enc_selfattn_3d = nn.ModuleList()
        self.enc_ffn_3d      = nn.ModuleList()
        self.num_layers_3d = cfg.ENTITY.FUSE_NUM_LAYERS
        for _ in range(self.num_layers_3d):
            self.enc_crosattn_3d.append(
                CrossAttentionLayer(
                    d_model=cfg.ENTITY.FUSE_ENC_HIDDIEN_DIM,
                    nhead=cfg.ENTITY.FUSE_ENC_NHEADS,
                    dropout=0.0,
                    normalize_before=cfg.ENTITY.FUSE_ENC_PRE_NORM)
            )
            self.enc_selfattn_3d.append(
                SelfAttentionLayer(
                    d_model=cfg.ENTITY.FUSE_ENC_HIDDIEN_DIM,
                    nhead=cfg.ENTITY.FUSE_ENC_NHEADS,
                    dropout=0.0,
                    normalize_before=cfg.ENTITY.FUSE_ENC_PRE_NORM)
                    )
            self.enc_ffn_3d.append(
                FFNLayer(
                    d_model=cfg.ENTITY.FUSE_ENC_HIDDIEN_DIM,
                    dim_feedforward=cfg.ENTITY.FUSE_ENC_DIM_FEEDFORWARD,
                    dropout=0.0,
                    normalize_before=cfg.ENTITY.FUSE_ENC_PRE_NORM,
                    )
            )
    
    def forward(self, output_2d, query_embed_2d, query_embed_3d):
        Q, BT, C = query_embed_2d.shape
        Q, B, C  = query_embed_3d.shape
        T = int(BT / B)

        output_3d = output_2d[:,0::T,:]
        ### (Q, B, T, C)
        output_2d = output_2d.unflatten(1, (B, T)).permute((0,2,1,3)).flatten(0,1)
        query_embed_2d = query_embed_2d.unflatten(1, (B, T)).permute((0,2,1,3)).flatten(0,1)

        for i in range(self.num_layers_3d):
            output_3d = self.enc_crosattn_3d[i](output_3d, output_2d, pos=query_embed_2d, query_pos=query_embed_3d)
            output_3d = self.enc_selfattn_3d[i](output_3d)
            output_3d = self.enc_ffn_3d[i](output_3d)
        
        return output_3d


@TRANSFORMER_DECODER_REGISTRY.register()
class CropSharedMultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        cfg,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.cfg = cfg

        self.mask_classification = mask_classification
        # positional encoding
        N_steps = hidden_dim // 2
        
        self.pe_layer = PositionEmbeddingSine3D2D(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.make_3d = Make3dQueries(cfg)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
                weight_init.c2_xavier_fill(self.input_proj_3d[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["cfg"] = cfg
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels

        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // (self.cfg.ENTITY.CROP_SAMPLE_NUM_TRAIN+1) if self.training else 1
        # bs = bt // self.num_views if self.training else 1
        t_m = bt // bs
        mask_features_2d = mask_features
        mask_features_3d = mask_features.view(bs, t_m, c_m, h_m, w_m)

        src_2d, src_3d = [], []
        pos_2d, pos_3d = [], []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        # pdb.set_trace()
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos_2d_, pos_3d_ = self.pe_layer(x[i].view(bs, t_m, -1, size_list[-1][0], size_list[-1][1]))
            
            pos_3d.append(pos_3d_.flatten(3))
            src_3d.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            pos_2d.append(pos_2d_.flatten(2))
            src_2d.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # NTxCxHW => NxTxCxHW => (TxHW)xNxC
            _, c, hw = src_3d[-1].shape
            pos_3d[-1] = pos_3d[-1].view(bs, t_m, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
            src_3d[-1] = src_3d[-1].view(bs, t_m, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

            pos_2d[-1] = pos_2d[-1].permute(2,0,1)
            src_2d[-1] = src_2d[-1].permute(2,0,1)

        # QxNxC
        query_embed_2d = self.query_embed.weight.unsqueeze(1).repeat(1, bt, 1)
        output_2d = self.query_feat.weight.unsqueeze(1).repeat(1, bt, 1)

        predictions_class_2d = []
        predictions_mask_2d  = []

        # prediction heads on learnable query features
        outputs_class_2d, outputs_mask_2d, attn_mask_2d, embedding_2d = self.forward_prediction_heads(output_2d, mask_features_2d, output_type="2d", attn_mask_target_size=size_list[0])
        predictions_class_2d.append(outputs_class_2d)
        predictions_mask_2d.append(outputs_mask_2d)
        
        # pdb.set_trace()
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask_2d[torch.where(attn_mask_2d.sum(-1) == attn_mask_2d.shape[-1])] = False
            # attention: cross-attention first
            output_2d = self.transformer_cross_attention_layers[i](
                output_2d, src_2d[level_index],
                memory_mask=attn_mask_2d,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos_2d[level_index], query_pos=query_embed_2d
            )

            output_2d = self.transformer_self_attention_layers[i](
                output_2d, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed_2d
            )
            
            # FFN
            output_2d = self.transformer_ffn_layers[i](
                output_2d
            )

            outputs_class_2d, outputs_mask_2d, attn_mask_2d, embedding_2d = self.forward_prediction_heads(output_2d, mask_features_2d, output_type="2d", attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class_2d.append(outputs_class_2d)
            predictions_mask_2d.append(outputs_mask_2d)

        assert len(predictions_class_2d) == self.num_layers + 1

        out_2d = {
            'pred_logits': predictions_class_2d[-1],
            'pred_masks': predictions_mask_2d[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class_2d if self.mask_classification else None, predictions_mask_2d
            )
        }

        predictions_class_3d = []
        predictions_mask_3d  = []

        query_embed_3d = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        output_3d = self.make_3d(output_2d, query_embed_2d, query_embed_3d)
        
        # self.fused
        outputs_class_3d, outputs_mask_3d, attn_mask_3d, embedding_3d = self.forward_prediction_heads(output_3d, mask_features_3d, output_type="3d", attn_mask_target_size=size_list[0])
        predictions_class_3d.append(outputs_class_3d)
        predictions_mask_3d.append(outputs_mask_3d)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask_3d[torch.where(attn_mask_3d.sum(-1) == attn_mask_3d.shape[-1])] = False
            ################# 3d (unified) #############
            # attention: cross-attention first
            output_3d = self.transformer_cross_attention_layers[i](
                output_3d, src_3d[level_index],
                memory_mask=attn_mask_3d,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos_3d[level_index], query_pos=query_embed_3d
            )

            output_3d = self.transformer_self_attention_layers[i](
                output_3d, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed_3d
            )
            
            output_3d = self.transformer_ffn_layers[i](
                output_3d
            )

            outputs_class_3d, outputs_mask_3d, attn_mask_3d, embedding_3d = self.forward_prediction_heads(output_3d, mask_features_3d, output_type="3d", attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class_3d.append(outputs_class_3d)
            predictions_mask_3d.append(outputs_mask_3d)

        # assert len(predictions_class_3d) == self.num_layers + 1

        out_3d = {
            'pred_logits': predictions_class_3d[-1],
            'pred_masks': predictions_mask_3d[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class_3d if self.mask_classification else None, predictions_mask_3d
            ),
        }

        return out_2d, out_3d

    def forward_prediction_heads(self, output, mask_features, output_type, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class  = self.class_embed(decoder_output)
        mask_embed     = self.mask_embed(decoder_output)
        if output_type == "3d":
            outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
            b, q, t, _, _ = outputs_mask.shape
            # NOTE: prediction is of higher-resolution
            # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
            attn_mask = F.interpolate(outputs_mask.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()
        elif output_type == "2d":
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()
        else:
            raise "the output_type should be 2d or 3d"
        
        return outputs_class, outputs_mask, attn_mask, decoder_output

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]