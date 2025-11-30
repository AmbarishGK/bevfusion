#!/usr/bin/env python
import argparse
import os

import torch
import mmcv

from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal BEVFusion inference (single sample, modern stack)"
    )
    parser.add_argument(
        "--config", required=True, help="Config file (.py or .yaml)"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Checkpoint file (.pth)"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index to run inference on (0-based)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
        help="Score threshold for printing boxes",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs (single process; >1 best-effort multi-GPU)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=None,
        help="Override cfg.data.workers_per_gpu (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed",
    )
    return parser.parse_args()


def setup_config(cfg_path):
    cfg = Config.fromfile(cfg_path)

    # disable any training init
    if hasattr(cfg, "model"):
        cfg.model.pretrained = None
        if hasattr(cfg.model, "init_cfg"):
            cfg.model.init_cfg = None
        cfg.model.train_cfg = None

    # enforce test_mode + samples_per_gpu=1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.samples_per_gpu = 1
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            ds_cfg.samples_per_gpu = 1

    if not hasattr(cfg.data, "workers_per_gpu"):
        cfg.data.workers_per_gpu = 2

    return cfg


def build_data(cfg, workers_per_gpu=None):
    dataset = build_dataset(cfg.data.test)

    if workers_per_gpu is None:
        workers_per_gpu = cfg.data.get("workers_per_gpu", 2)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    return dataset, data_loader


def build_and_load_model(cfg, checkpoint_path, device, num_gpus):
    device_is_cuda = (device == "cuda" and torch.cuda.is_available())
    if device_is_cuda and num_gpus > torch.cuda.device_count():
        num_gpus = torch.cuda.device_count()

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    # mmcv.runner.load_checkpoint is deprecated in mmcv>=2; use mmengine.runner.load_checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")

    if "meta" in checkpoint and "CLASSES" in checkpoint["meta"]:
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = getattr(model, "CLASSES", None)

    model.eval()

    # single-/multi-GPU
    if device_is_cuda:
        model = model.cuda()
        if num_gpus > 1:
            # MMDataParallel was removed from mmcv>=2 for many envs; fall back to torch.nn.DataParallel.
            # This is best-effort for inference; if it breaks, use --num-gpus 1.
            device_ids = list(range(num_gpus))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model = model.cpu()

    return model


def fetch_batch(data_loader, index):
    if index < 0:
        raise ValueError("--index must be >= 0")

    for i, data in enumerate(data_loader):
        if i == index:
            return data

    raise IndexError(
        f"Requested index {index}, but dataset has only {i + 1} samples."
    )


def decode_and_print(result, model, dataset, score_thresh=0.0):
    # mmdet3d typically returns a list (batch), even with batch size 1
    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            print("Empty result list.")
            return
        result = result[0]

    # BEVFusion-style detectors usually store 3D boxes under 'pts_bbox'
    if isinstance(result, dict) and "pts_bbox" in result:
        det = result["pts_bbox"]
    else:
        det = result

    boxes_3d = det.get("boxes_3d", None)
    scores_3d = det.get("scores_3d", None)
    labels_3d = det.get("labels_3d", None)

    if boxes_3d is None or scores_3d is None or labels_3d is None:
        print("Result does not contain boxes_3d / scores_3d / labels_3d.")
        print(f"Keys: {list(det.keys()) if isinstance(det, dict) else type(det)}")
        return

    # boxes_3d.tensor: (N, 7) = x,y,z,dx,dy,dz,yaw in LiDAR coords (nuScenes-style).
    boxes = boxes_3d.tensor.detach().cpu().numpy()
    scores = scores_3d.detach().cpu().numpy()
    labels = labels_3d.detach().cpu().numpy()

    classes = getattr(model, "CLASSES", None)
    if classes is None and hasattr(dataset, "CLASSES"):
        classes = dataset.CLASSES

    print(f"\nDetected 3D boxes (score >= {score_thresh}):")
    print("idx | class        | score   | x      y      z      dx     dy     dz     yaw")
    print("-" * 90)
    for i in range(len(scores)):
        if scores[i] < score_thresh:
            continue

        x, y, z, dx, dy, dz, yaw = boxes[i].tolist()
        label_id = int(labels[i])
        if classes is not None and 0 <= label_id < len(classes):
            cls_name = classes[label_id]
        else:
            cls_name = str(label_id)

        print(
            f"{i:3d} | {cls_name:11s} | {scores[i]:6.3f} | "
            f"{x:6.2f} {y:6.2f} {z:6.2f} "
            f"{dx:6.2f} {dy:6.2f} {dz:6.2f} {yaw:6.3f}"
        )


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cfg = setup_config(args.config)
    dataset, data_loader = build_data(cfg, workers_per_gpu=args.workers_per_gpu)

    print(f"Test dataset length: {len(dataset)}")
    print(f"Using sample index: {args.index}")

    batch = fetch_batch(data_loader, args.index)

    model = build_and_load_model(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_gpus=args.num_gpus,
    )

    with torch.no_grad():
        outputs = model(return_loss=False, rescale=True, **batch)

    decode_and_print(outputs, model, dataset, score_thresh=args.score_thresh)


if __name__ == "__main__":
    main()
