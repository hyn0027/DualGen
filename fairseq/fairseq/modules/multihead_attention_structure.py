# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

try:
    from xformers.components.attention import build_attention
    from xformers.components.attention.utils import maybe_merge_masks

    _xformers_available = True
except ImportError:
    _xformers_available = False

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder


# TODO: move this into xformers?
# TODO: uint8 input type should just output a bool
def _mask_for_xformers(mask: Tensor, to_dtype: Optional[torch.dtype] = None):
    """
    call to pytorch multihead accepts three mask types:
        - ByteTensor where non-zero means to mask
        - FloatTensor which is an additive mask
        - BoolTensor where True means to mask
    xFormers currently accepts boolean and additive maks. For boolean masks
    the values have opposite meaning. For a BoolTensor True mean to keep the value.
    """
    float_types = [torch.float, torch.float16]
    # If an input mask is a float it is an additive mask. Otherwise it is either uint8 or bool.
    additive = mask.dtype in float_types
    # If to_dype is not specified, keep same dtype as mask.
    to_dtype = mask.dtype if to_dtype is None else to_dtype
    to_additive = to_dtype in float_types

    if additive:
        if to_additive:
            return mask.to(to_dtype)
        mask = mask < 0

    if to_additive:
        # return additive mask
        new_mask = torch.zeros_like(mask, dtype=to_dtype)
        new_mask = new_mask.masked_fill_(mask, -float("inf"))
        return new_mask

    # In xFormers True is value to keep rather than value to mask
    mask = ~mask.to(torch.bool)
    mask = mask.to(to_dtype)
    return mask


class StructureMultiheadAttention(FairseqIncrementalDecoder):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        dictionary=None,
        q_noise=0.0,
        qn_block_size=8,
        # TODO: pass in config rather than string.
        # config defined in xformers.components.attention.AttentionConfig
        xformers_att_config: Optional[str] = None,
        xformers_blocksparse_layout: Optional[
            torch.Tensor
        ] = None,  # This should be part of the config
        xformers_blocksparse_blocksize: Optional[
            int
        ] = 16,  # This should be part of the config
    ):
        super().__init__(dictionary)

        xformers_att_config = utils.eval_str_dict(xformers_att_config)
        self.use_xformers = xformers_att_config is not None
        if self.use_xformers and not _xformers_available:
            raise ImportError("\n\n  Please install xFormers.")
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.beam_size = 1
        self.reset_parameters()

        if self.use_xformers:
            xformers_att_config["dropout"] = xformers_att_config.get("dropout", dropout)
            xformers_att_config["num_heads"] = xformers_att_config.get(
                "num_heads", num_heads
            )

            if xformers_blocksparse_layout is not None:
                # Could be part of a single config passed only once
                xformers_att_config["block_size"] = xformers_blocksparse_blocksize
                xformers_att_config["layout"] = xformers_blocksparse_layout
                xformers_att_config["name"] = "blocksparse"

            self.attention = build_attention(xformers_att_config)

        self.onnx_trace = False
        self.skip_embed_dim_check = False
        self.init_incremental_state()

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _get_reserve_head_index(self, num_heads_to_keep: int):
        k_proj_heads_norm = []
        q_proj_heads_norm = []
        v_proj_heads_norm = []

        for i in range(self.num_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        self.k_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(self.k_proj.bias[start_idx:end_idx])).tolist()
            )
            q_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        self.q_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(self.q_proj.bias[start_idx:end_idx])).tolist()
            )
            v_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        self.v_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(self.v_proj.bias[start_idx:end_idx])).tolist()
            )

        heads_norm = []
        for i in range(self.num_heads):
            heads_norm.append(
                k_proj_heads_norm[i] + q_proj_heads_norm[i] + v_proj_heads_norm[i]
            )

        sorted_head_index = sorted(
            range(self.num_heads), key=lambda k: heads_norm[k], reverse=True
        )
        reserve_head_index = []
        for i in range(num_heads_to_keep):
            start = sorted_head_index[i] * self.head_dim
            end = (sorted_head_index[i] + 1) * self.head_dim
            reserve_head_index.append((start, end))
        return reserve_head_index

    def _adaptive_prune_heads(self, reserve_head_index: List[Tuple[int, int]]):
        new_q_weight = []
        new_q_bias = []
        new_k_weight = []
        new_k_bias = []
        new_v_weight = []
        new_v_bias = []
        new_out_proj_weight = []

        for ele in reserve_head_index:
            start_idx, end_idx = ele
            new_q_weight.append(
                self.q_proj.weight[
                    start_idx:end_idx,
                ]
            )
            new_q_bias.append(self.q_proj.bias[start_idx:end_idx])

            new_k_weight.append(
                self.k_proj.weight[
                    start_idx:end_idx,
                ]
            )

            new_k_bias.append(self.k_proj.bias[start_idx:end_idx])

            new_v_weight.append(
                self.v_proj.weight[
                    start_idx:end_idx,
                ]
            )
            new_v_bias.append(self.v_proj.bias[start_idx:end_idx])

            new_out_proj_weight.append(self.out_proj.weight[:, start_idx:end_idx])

        new_q_weight = torch.cat(new_q_weight).detach()
        new_k_weight = torch.cat(new_k_weight).detach()
        new_v_weight = torch.cat(new_v_weight).detach()
        new_out_proj_weight = torch.cat(new_out_proj_weight, dim=-1).detach()
        new_q_weight.requires_grad = True
        new_k_weight.requires_grad = True
        new_v_weight.requires_grad = True
        new_out_proj_weight.requires_grad = True

        new_q_bias = torch.cat(new_q_bias).detach()
        new_q_bias.requires_grad = True

        new_k_bias = torch.cat(new_k_bias).detach()
        new_k_bias.requires_grad = True

        new_v_bias = torch.cat(new_v_bias).detach()
        new_v_bias.requires_grad = True

        self.q_proj.weight = torch.nn.Parameter(new_q_weight)
        self.q_proj.bias = torch.nn.Parameter(new_q_bias)

        self.k_proj.weight = torch.nn.Parameter(new_k_weight)
        self.k_proj.bias = torch.nn.Parameter(new_k_bias)

        self.v_proj.weight = torch.nn.Parameter(new_v_weight)
        self.v_proj.bias = torch.nn.Parameter(new_v_bias)

        self.out_proj.weight = torch.nn.Parameter(new_out_proj_weight)

        self.num_heads = len(reserve_head_index)
        self.embed_dim = self.head_dim * self.num_heads
        self.q_proj.out_features = self.embed_dim
        self.k_proj.out_features = self.embed_dim
        self.v_proj.out_features = self.embed_dim

    def _set_skip_embed_dim_check(self):
        self.skip_embed_dim_check = True

    def _pad_masks(
        self,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if attn_mask is not None:
            shape = attn_mask.size()[:-1] + torch.Size([1])
            attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(shape)], dim=-1)
        if key_padding_mask is not None:
            shape = key_padding_mask.size()[:-1] + torch.Size([1])
            key_padding_mask = torch.cat(
                [
                    key_padding_mask,
                    key_padding_mask.new_zeros(shape),
                ],
                dim=-1,
            )
        return key_padding_mask, attn_mask

    def _add_bias(
        self,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
        bsz: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        assert self.bias_k is not None
        assert self.bias_v is not None
        k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        key_padding_mask, attn_mask = self._pad_masks(
            key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        return k, v, key_padding_mask, attn_mask

    def _append_zero_attn(
        self,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        zero_attn_shape = k.size()[:-2] + torch.Size([1]) + k.size()[-1:]
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=-2
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=-2
        )
        key_padding_mask, attn_mask = self._pad_masks(
            key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        return k, v, key_padding_mask, attn_mask

    def _xformers_attn_forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        tgt_len, bsz, embed_dim = query.size()

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == tgt_len

        if self.self_attention:
            key = query
            value = query
        elif self.encoder_decoder_attention:
            value = key

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k, v, attn_mask, key_padding_mask, bsz
            )

        def fold_heads(x):
            return (
                x.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        def split_heads(x):
            return (
                x.contiguous()
                .view(-1, bsz, self.num_heads, self.head_dim)
                .transpose(0, 1)
                .transpose(1, 2)
            )

        massage = split_heads if self.attention.requires_head_dimension else fold_heads
        q = massage(q)
        if k is not None:
            k = massage(k)
        if v is not None:
            v = massage(v)

        if self.add_zero_attn:
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        kwargs = {}

        if attn_mask is not None and self.attention.supports_attention_mask:
            attn_mask = _mask_for_xformers(attn_mask, to_dtype=q.dtype)
            kwargs["att_mask"] = attn_mask

        if key_padding_mask is not None:
            key_padding_mask = _mask_for_xformers(key_padding_mask, to_dtype=torch.bool)
            if not self.attention.requires_separate_masks:
                attn_mask = maybe_merge_masks(
                    attn_mask,
                    key_padding_mask,
                    batch_size=bsz,
                    src_len=k.size(-2),
                    tgt_len=q.size(-2),
                    num_heads=self.num_heads,
                )
                key_padding_mask = None
                kwargs["att_mask"] = attn_mask
            if self.attention.supports_key_padding_mask:
                kwargs["key_padding_mask"] = key_padding_mask

        y = self.attention(q, k, v, **kwargs)

        y = (
            y.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .flatten(start_dim=2, end_dim=3)
            .transpose(0, 1)
        )
        assert list(y.size()) == [tgt_len, bsz, embed_dim]

        # Dropout not needed because already applied in attention.
        # It is applied to the attention weights before matmul with v.
        y = self.out_proj(y)

        # TODO: support returning attention weights if needed.
        return y, None

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        structure: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
            # The Multihead attention implemented in pytorch forces strong dimension check
            # for input embedding dimention and K,Q,V projection dimension.
            # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
            # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
            and not self.skip_embed_dim_check
        ):
            assert key is not None and value is not None

            if self.use_xformers:
                print(1)
                exit(-1)
                return self._xformers_attn_forward(
                    query, key, value, key_padding_mask, need_weights, attn_mask
                )

            else:
                # print(2)
                # exit(-1)
                return F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout_module.p,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training or self.dropout_module.apply_during_inference,
                    key_padding_mask.bool() if key_padding_mask is not None else None,
                    need_weights,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                )
        print("not implemented! multihead attention sturcture")
        exit(-2)