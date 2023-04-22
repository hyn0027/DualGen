# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import utils, options
from fairseq.logging import metrics
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)
def load_amr_dataset(
    data_path, split,
    add_dict,
    dataset_impl, combine, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False, num_buckets=0, 
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    # ------------load dataset start--------------------------

    # infer langcode
    if split_exists(split, "source", "target", "source", data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, "source", "target"))
    else:
        raise FileNotFoundError(
            'Dataset not found: {} ({})'.format(split, data_path)
        )

    # load src
    src_dataset = data_utils.load_indexed_dataset(
        prefix + "source", add_dict, dataset_impl
    )
    if truncate_source:
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, add_dict.eos()),
                max_source_positions - 1,
            ),
            add_dict.eos(),
        )

    # load tgt
    tgt_dataset = data_utils.load_indexed_dataset(
        prefix + "target", add_dict, dataset_impl
    )
    
    # load graph info
    prefix = os.path.join(data_path, "{}.{}-{}.".format(split, "info", "None"))
    graphInfo = data_utils.load_indexed_dataset(prefix + "info", add_dict, dataset_impl)
    # graphInfo need to - 3
    # graph_structure = []
    # for item in graphInfo:
    #     # item = item.split()
    #     graph_structure.append({})
    #     graph_structure[-1]["node_num"] = int(item[0])
    #     graph_structure[-1]["root"] = int(item[1])
    #     graph_structure[-1]["edge"] = []
    #     i = 2
    #     while i < len(item):
    #         graph_structure[-1]["edge"].append(torch.tensor([int(item[i]), int(item[i + 1])], dtype=torch.long))
    #         i += 2
    #     if len(graph_structure[-1]["edge"]) > 0:
    #         graph_structure[-1]["edge"] = torch.stack(graph_structure[-1]["edge"])
    #         graph_structure[-1]["edge"] = graph_structure[-1]["edge"].transpose(0, 1)
    #     else:
    #         graph_structure[-1]["edge"] = torch.tensor([[], []], dtype=torch.long)
    
    # edge_dataset = data_utils.load_indexed_dataset(prefix + "edge", add_dict, dataset_impl)
    # # load node
    # node_list = data_utils.load_indexed_dataset(prefix + "node", add_dict, dataset_impl)
    # cnt = 0
    # node_dataset = []
    # for item in graph_structure:
    #     node_dataset.append([])
    #     for j in range(item["node_num"]):
    #         node_dataset[-1].append(node_list[cnt][:-1])
    #         cnt += 1
    #     cnt += 1

    logger.info('{} {} {} examples'.format(
        data_path, split, len(src_dataset)
    ))

    # ------------load dataset end--------------------------

    assert len(src_dataset) == len(tgt_dataset)
    # assert len(src_dataset) == len(graph_structure)
    # assert len(src_dataset) == len(edge_dataset)
    # assert len(src_dataset) == len(node_dataset)
    if prepend_bos:
        assert hasattr(add_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, add_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, add_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)
    
    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, add_dict.index("[{}]".format("source"))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, add_dict.index("[{}]".format("target"))
            )
        eos = add_dict.index("[{}]".format("target"))
    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, "source", "target"))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    return None

@register_task('graph_to_seq')
class GraphToSeq(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', default=None, help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--source_lang', default=None, help="source language")
        parser.add_argument('--target_lang', default=None, help="target language")
        parser.add_argument('--num_batch_buckets', default=0, type=int, help="if >0, then bucket source and target lengths into N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations")
        parser.add_argument('--load_alignments', default=False, type=bool, help="load the binarized alignments")
        parser.add_argument('--left-pad-source', default=True, type=bool, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default=False, type=bool, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--upsample-primary', default=-1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--eval-bleu', default=False, action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenizer before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar="{}",
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-bleu-args', type=str, metavar="{}",
                            help='generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='if setting, we compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-print-samples', action='store_true', default=False,
                            help='print sample generations during validation')
        
    def __init__(self, args, add_dict):
        super().__init__(args)
        self.add_dict = add_dict
        self.args=args

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        add_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.source.txt'))
        logger.info('adding dictionary: {} types'.format(len(add_dict)))
        return cls(args, add_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        self.datasets[split] = load_amr_dataset(
            data_path, split, self.add_dict,
            dataset_impl=self.args.dataset_impl,
            combine=combine,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )
    
    def build_dataset_for_inference(self, src_tokens, src_lengths, graph_structure, edge, node):
        pass

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        if getattr(args, 'eval_bleu', True):
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                import torch
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=int(meters["_bleu_sys_len"].sum),
                        ref_len=int(meters["_bleu_ref_len"].sum),
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.add_dict
    
    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.add_dict
    
    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.add_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(
                decode(
                    utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])