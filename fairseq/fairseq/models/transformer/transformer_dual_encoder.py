# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)

import torch.nn as nn
from fairseq.models.transformer import TransformerEncoderBase

from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,
)
from typing import Optional
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap

from fairseq.modules import (
    LayerDropModuleList,
    transformer_layer,
)
import torch
import torch.nn.functional as F

@register_model("transformerDualEnc")
class TransformerDualEncModel(TransformerModelBase):
    """
    This is the legacy implementation of the transformer model that
    uses argparse for configuration.
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        def spm(path):
            return {
                'path': path,
                'bpe': 'sentencepiece',
                'tokenizer': 'space',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
            'transformer.wmt20.en-ta': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-ta.single.tar.gz'),
            'transformer.wmt20.en-iu.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.news.single.tar.gz'),
            'transformer.wmt20.en-iu.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.nh.single.tar.gz'),
            'transformer.wmt20.ta-en': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta-en.single.tar.gz'),
            'transformer.wmt20.iu-en.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.news.single.tar.gz'),
            'transformer.wmt20.iu-en.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.nh.single.tar.gz'),
            'transformer.flores101.mm100.615M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz'),
            'transformer.flores101.mm100.175M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )
        parser.add_argument(
            '--freeze', type=int, default=0,
            help="freeze everything except graph encoder"
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            TransformerConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return GraphToTextEncoder(TransformerConfig.from_namespace(args), src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return super().build_decoder(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        graph_structures,
        nodes,
        nodes_info,
        edges,
        edges_info,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, 
            graph_structures=graph_structures,
            nodes=nodes,
            nodes_info=nodes_info,
            edges=edges,
            edges_info=edges_info,
            return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths, # TODO 是否需要修改
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out
    
class GraphToTextEncoder(TransformerEncoderBase):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """
    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        super().__init__(cfg, dictionary, embed_tokens, return_fc)
        if self.encoder_layerdrop > 0.0:
            self.graph_layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.graph_layers = nn.ModuleList([])
        self.graph_layers.extend(
            [self.build_graph_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.graph_embeddings = nn.Linear(1024, 64)
        self.graph_embeddings_inverse = nn.Linear(1024, 64)
        self.gamma_norm = torch.nn.LayerNorm(64)
    
    def build_graph_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerGraphEncoderLayerBase(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        graph_structures = None,
        nodes = None,
        nodes_info = None,
        edges = None,
        edges_info = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if token_embeddings != None:
            print("not implemented at /home/hongyining/s_link/dualEnc_virtual/fairseq/fairseq/models/transformer/transformer_dual_encoder.py, forward")
            exit(-2)
        # s2s_output = self.forward_scriptable(
        #     self.graph_layers, src_tokens, src_lengths, return_all_hiddens, token_embeddings
        # )
        # s2s_output = self.forward_scriptable(
        #     self.layers, src_tokens, src_lengths, return_all_hiddens, token_embeddings
        # )

        # print(src_tokens.size())
        g2s_output = self.graph_foward_scriptable(self.graph_layers, graph_structures, nodes, nodes_info, edges, edges_info, return_all_hiddens)
        return g2s_output
        # return s2s_output
        # return self.combine_results(s2s_output, g2s_output) # TODO

    def dfs(self, p, graph_structure):
        res = [{"type": "node", "id": p}]
        if self.visited[int(p)]:
            return res
        self.visited[p] = True
        for i in range(int(len(graph_structure) / 2)):
            s = graph_structure[2 * i]
            e = graph_structure[2 * i + 1]
            if s == p:
                child = self.dfs(e, graph_structure)
                if len(child) > 1:
                    child = [{"type": "l"}] + child + [{"type": "r"}]
                res = res + [{"type": "edge", "id": i}] + child
        return res

    def graph_forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed
    
    def embedding_gamma(self, gamma):
        gamma = self.embed_scale * gamma
        gamma = self.gamma_norm(gamma)
        gamma = self.dropout_module(gamma)
        return gamma

    def graph_foward_scriptable(
        self,
        layers,
        graph_structures, 
        nodes, nodes_info, 
        edges, edges_info,
        return_all_hiddens: bool = False,
    ):
        batch_size = len(graph_structures)
        token_embeddings = []
        gamma = []
        max_length = 0
        for i in range(batch_size):
            token_embeddings_nodes_tmp = self.embed_tokens(nodes[i][None, :])
            token_embeddings_edges_tmp = self.embed_tokens(edges[i][None, :])
            token_embeddings_nodes_tmp = token_embeddings_nodes_tmp[0]
            token_embeddings_edges_tmp = token_embeddings_edges_tmp[0]
            token_embeddings_nodes = []
            prev_idx = 0
            for idx in range(len(nodes_info[i]) - 1):
                num_node_tokens = int(nodes_info[i][idx])
                token_embeddings_nodes.append(token_embeddings_nodes_tmp[prev_idx: prev_idx + num_node_tokens].mean(0))
                prev_idx += num_node_tokens
            token_embeddings_nodes = torch.stack(token_embeddings_nodes)

            token_embeddings_edges = []
            token_embeddings_edges_inverse = []
            prev_idx = 0
            for idx in range(len(edges_info[i]) - 1):
                num_edge_tokens = int(edges_info[i][idx])
                token_embeddings_edges.append(self.graph_embeddings(token_embeddings_edges_tmp[prev_idx + 1: prev_idx + num_edge_tokens].mean(0)))
                token_embeddings_edges_inverse.append(self.graph_embeddings_inverse(token_embeddings_edges_tmp[prev_idx + 1: prev_idx + num_edge_tokens].mean(0)))
                # token_embeddings_edges.append(token_embeddings_edges_tmp[prev_idx: prev_idx + num_edge_tokens].mean(0))
                prev_idx += num_edge_tokens
            if len(token_embeddings_edges) > 0:
                token_embeddings_edges = torch.stack(token_embeddings_edges)
                token_embeddings_edges_inverse = torch.stack(token_embeddings_edges_inverse)
            else:
                token_embeddings_edges = torch.ones((0, 64), device=token_embeddings_nodes.get_device(), dtype=token_embeddings_nodes.dtype)
                token_embeddings_edges_inverse = torch.ones((0, 64), device=token_embeddings_nodes.get_device(), dtype=token_embeddings_nodes.dtype)

            gamma_single = torch.zeros(
                [len(nodes_info[i]) - 1, len(nodes_info[i]) - 1, 64],
                device=token_embeddings_nodes.get_device(),
                dtype=token_embeddings_nodes.dtype
            )
            for idx in range(len(edges_info[i]) - 1):
                edge_i = graph_structures[i][2 + idx * 2]
                edge_j = graph_structures[i][2 + idx * 2 + 1]
                gamma_single[edge_i][edge_j] = token_embeddings_edges[idx]
                gamma_single[edge_j][edge_i] = token_embeddings_edges_inverse[idx]
            # self.visited = [False for _i in range(int(graph_structures[i][0]))]
            # dfs_path = self.dfs(graph_structures[i][1], graph_structures[i][2: -1])
            # token_embeddings_single = []
            # for item in dfs_path:
            #     if item["type"] == "node":
            #         token_embeddings_single.append(token_embeddings_nodes[int(item["id"])])
            #     elif item["type"] == "edge":
            #         token_embeddings_single.append(token_embeddings_edges[int(item["id"])])
            #     elif item["type"] == "l":
            #         token_embeddings_single.append(self.embed_tokens(torch.tensor([[36]], device=nodes[i].get_device()))[0][0])
            #     elif item["type"] == "r":
            #         token_embeddings_single.append(self.embed_tokens(torch.tensor([[4839]], device=nodes[i].get_device()))[0][0])
            # token_embeddings_single.append(self.embed_tokens(torch.tensor([[2]], device=nodes[i].get_device()))[0][0])
            # token_embeddings_single = torch.stack(token_embeddings_single)
            token_embeddings_single = token_embeddings_nodes
            token_embeddings.append(token_embeddings_single)
            gamma.append(gamma_single)
            max_length = max(max_length, token_embeddings_single.size(0))
        encoder_padding_mask = []
        src_tokens = []
        for idx in range(batch_size):
            encoder_padding_mask.append(
                torch.tensor(
                    [False for _i in range(token_embeddings[idx].size(0))]
                    + [True for _i in range(token_embeddings[idx].size(0), max_length)], 
                    device=token_embeddings[idx].get_device(),
                    dtype=torch.bool
                )
            )
            src_tokens.append(
                torch.tensor(
                    [10 for _i in range(token_embeddings[idx].size(0))]
                    + [self.padding_idx for _i in range(token_embeddings[idx].size(0), max_length)], 
                    device=token_embeddings[idx].get_device(),
                    dtype=torch.long
                )
            )
            padding_token = self.embed_tokens(
                torch.tensor(
                    [[self.padding_idx for _i in range(token_embeddings[idx].size(0), max_length)]],
                    device=token_embeddings[idx].get_device(),
                    dtype=torch.long,
                )
            )[0]
            token_embeddings[idx] = torch.cat((token_embeddings[idx], padding_token), 0)

            additional_dim = max_length - gamma[idx].size(0)
            gamma[idx] = F.pad(gamma[idx], (0, 0, additional_dim, 0, additional_dim, 0), "constant", 0)
        token_embeddings = torch.stack(token_embeddings)
        encoder_padding_mask = torch.stack(encoder_padding_mask)
        src_tokens = torch.stack(src_tokens)
        gamma = torch.stack(gamma, dim=2)
        gamma = self.embedding_gamma(gamma)
        has_pads = (
            torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        )
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        x, encoder_embedding = self.graph_forward_embedding(src_tokens, token_embeddings)
        # account for padding while computing the representation
        x = x * (
            1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None, gamma=gamma
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    def forward_scriptable(
        self,
        layers,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (
            torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        )
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        
        # account for padding while computing the representation
        x = x * (
            1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }
    
    def combine_results(self, s2soutput, g2soutput):
        s2s_encoder_out = s2soutput["encoder_out"][0]
        s2s_encoder_padding_mask = s2soutput["encoder_padding_mask"][0]
        s2s_encoder_embedding = s2soutput["encoder_embedding"][0]
        s2s_encoder_states = s2soutput["encoder_states"]
        s2s_fc_results = s2soutput["fc_results"]
        s2s_src_lengths = s2soutput["src_lengths"][0]

        g2s_encoder_out = g2soutput["encoder_out"][0]
        g2s_encoder_padding_mask = g2soutput["encoder_padding_mask"][0]
        g2s_encoder_embedding = g2soutput["encoder_embedding"][0]
        g2s_encoder_states = g2soutput["encoder_states"]
        g2s_fc_results = g2soutput["fc_results"]
        g2s_src_lengths = g2soutput["src_lengths"][0]

        
        encoder_out = torch.cat((g2s_encoder_out, s2s_encoder_out), dim=0)
        encoder_padding_mask = torch.cat((g2s_encoder_padding_mask, s2s_encoder_padding_mask), dim=1)
        encoder_embedding = torch.cat((g2s_encoder_embedding, s2s_encoder_embedding), dim=1)
        assert len(s2s_encoder_states) == len(g2s_encoder_states)
        encoder_states = []
        for i in range(len(s2s_encoder_states)):
            encoder_states.append(
                torch.cat(
                    (s2s_encoder_states[i], g2s_encoder_states[i]),
                    dim=0
                )
            )

        fc_results = []
        for i in range(len(s2s_fc_results)):
            if s2s_fc_results[i] != None:
                fc_results.append(
                    torch.cat(
                        (s2s_fc_results[i], g2s_fc_results[i]),
                        dim=0
                    )
                )
            else:
                fc_results.append(None)
        src_lengths = s2s_src_lengths + g2s_src_lengths
        return {
            "encoder_out": [encoder_out],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

# architectures


@register_model_architecture("transformerDualEnc", "transformerDualEnc_tiny")
def tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    return base_architecture(args)


@register_model_architecture("transformerDualEnc", "transformerDualEnc")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.merge_src_tgt_embed = getattr(args, "merge_src_tgt_embed", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("transformerDualEnc", "transformerDualEnc_iwslt_de_en")
def transformerDualEnc_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformerDualEnc", "transformerDualEnc_wmt_en_de")
def transformerDualEnc_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformerDualEnc", "transformerDualEnc_vaswani_wmt_en_de_big")
def transformerDualEnc_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformerDualEnc", "transformerDualEnc_vaswani_wmt_en_fr_big")
def transformerDualEnc_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformerDualEnc_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformerDualEnc", "transformerDualEnc_wmt_en_de_big")
def transformerDualEnc_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformerDualEnc_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformerDualEnc", "transformerDualEnc_wmt_en_de_big_t2t")
def transformerDualEnc_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformerDualEnc_vaswani_wmt_en_de_big(args)
