#!/usr/bin/env python
import os
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)

from zsis.config import get_cfg
from zsis.data import DatasetMapper
from zsis.modeling import GeneralizedRCNNClip  # NOQA: F401


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        assert cfg.SOLVER.NAME in ['SGD', 'Adam'], f'Unsupported Solver {cfg.SOLVER.NAME}'
        if cfg.SOLVER.NAME == 'SGD':
            return super().build_optimizer(cfg, model)

        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )

        optim_args = {
            'params': params,
            'lr': cfg.SOLVER.BASE_LR,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
        }
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam(**optim_args))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ['sem_seg', 'coco_panoptic_seg']:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ['coco', 'coco_panoptic_seg']:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == 'coco_panoptic_seg':
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                'no Evaluator for the dataset {} with the type {}'.format(dataset_name, evaluator_type)
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.name is not None:
        cfg.DATASETS.TRAIN = [args.name]

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.name is not None:
        json_file = os.path.join(args.root, 'trainval.json')
        DatasetCatalog.register(args.name, lambda: load_coco_json(json_file, args.root, args.name))
        MetadataCatalog.get(args.name).set(json_file=json_file, image_root=args.root, evaluator_type='coco', **{})

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))])
    return trainer.train()


def get_argument_parser():
    parser = default_argument_parser()
    parser.add_argument('--root', required=False, type=str)
    parser.add_argument('--json', required=False, default=None, type=str)
    parser.add_argument('--name', required=False, default=None, type=str)
    return parser


if __name__ == '__main__':
    args = get_argument_parser().parse_args()

    print(f'\033[32m \033[5mCommand Line Args: \033[25m{args}\033[0m')
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
