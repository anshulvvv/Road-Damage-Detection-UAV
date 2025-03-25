import os
import torch
import cv2
import json
import argparse
import numpy as np

# Import MMDetection inference APIs
from mmdet.apis import init_detector, inference_detector


# Training configuration defaults

DEFAULT_BATCH_SIZE = 2
DEFAULT_EPOCHS = 1
DEFAULT_DATA_ROOT = "C:/Users/ANSHUL M/Downloads/RDD2022_India/India/"

def create_custom_config(data_root, batch_size, epochs):
    """
    Create a custom configuration file for RTMDet training.
    
    Args:
        data_root (str): Root directory for the training data.
        batch_size (int): Batch size per GPU.
        epochs (int): Maximum number of training epochs.
        
    Returns:
        str: The path to the custom configuration file.
    """
    HOME = os.getcwd()
    config_dir = os.path.join(HOME, "configs", "rtmdet")
    os.makedirs(config_dir, exist_ok=True)
    custom_config_path = os.path.join(config_dir, "custom.py")
    
    CUSTOM_CONFIG = f"""
_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# ---------------- Data Settings ----------------
data_root = "{data_root}"

train_ann_file = "train/annotations/annotations_coco.json"
train_data_prefix = "train/images"

val_ann_file = "val/annotations/annotations_coco.json"
val_data_prefix = "val/images"

num_classes = 4
class_name = ['D00', 'D10', 'D20', 'D40']
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

train_batch_size_per_gpu = {batch_size}
train_num_workers = 4
persistent_workers = True

# ---------------- Training Settings ----------------
base_lr = 0.004
max_epochs = {epochs}
num_epochs_stage2 = 20

model_test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.65),
    max_per_img=300)

# ---------------- Image and Pipeline Settings ----------------
img_scale = (640, 640)
random_resize_ratio_range = (0.1, 2.0)
mosaic_max_cached_images = 40
mixup_max_cached_images = 20
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 32
val_num_workers = 10

batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

deepen_factor = 1.0
widen_factor = 1.0
strides = [8, 16, 32]
norm_cfg = dict(type='BN')

lr_start_factor = 1.0e-5
dsl_topk = 13
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0
weight_decay = 0.05

save_checkpoint_intervals = 10
val_interval_stage2 = 1
max_keep_ckpts = 3
env_cfg = dict(cudnn_benchmark=True)

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),]

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='CSPNeXtPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetHead',
        head_module=dict(
            type='RTMDetSepBNHeadModule',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=qfl_beta,
            loss_weight=loss_cls_weight),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=loss_bbox_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_classes,
            topk=dsl_topk,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=model_test_cfg,
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=mosaic_max_cached_images,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=mixup_max_cached_images),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_start_factor,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts
    ))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals,
    dynamic_intervals=[(max_epochs - num_epochs_stage2, val_interval_stage2)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
    """
    
    with open(custom_config_path, "w") as f:
        f.write(CUSTOM_CONFIG)
    print(f"Custom config created at: {custom_config_path}")
    return custom_config_path

def train_model(custom_config_path, resume_checkpoint=None):
    
    # If a resume checkpoint is provided, add it to the command.
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        train_cmd = f"python tools/train.py {custom_config_path} --resume-from {resume_checkpoint}"
    else:
        train_cmd = f"python tools/train.py {custom_config_path}"
    print("Starting training with command:", train_cmd)
    os.system(train_cmd)
    # Assume the checkpoint is saved in work_dirs/custom/latest.pth.
    checkpoint_path = os.path.join("work_dirs", "custom", "latest.pth")
    return checkpoint_path

def pseudo_label_inference(model, test_dir, output_json):
    
    pseudo_labels = {}
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        result = inference_detector(model, img_path)
        boxes = []
        for class_idx, class_boxes in enumerate(result):
            for box in class_boxes:
                score = box[-1]
                if score >= 0.001:  # Confidence threshold
                    boxes.append({
                        "label": class_idx,
                        "bbox": box[:4].tolist(),
                        "score": float(score)
                    })
        pseudo_labels[img_file] = boxes
    with open(output_json, "w") as f:
        json.dump(pseudo_labels, f, indent=2)
    print("Pseudo labels saved to", output_json)

def run_training_and_inference(test_dir,
                               output_json="pseudo_labels.json",
                               data_root=DEFAULT_DATA_ROOT,
                               batch_size=DEFAULT_BATCH_SIZE,
                               epochs=DEFAULT_EPOCHS,
                               resume_checkpoint=None):
    
    # Step 1: Create custom config and train the model.
    config_path = create_custom_config(data_root, batch_size, epochs)
    checkpoint_path = train_model(config_path, resume_checkpoint)
    
    # Step 2: Initialize the model for inference.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_detector(config_path, checkpoint_path, device=device)
    
    # Step 3: Run pseudo label inference on the test set.
    pseudo_label_inference(model, test_dir, output_json)

def parse_args():
    parser = argparse.ArgumentParser(description="Train RTMDet and generate pseudo labels on test set")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test set images")
    parser.add_argument("--output_json", type=str, default="pseudo_labels.json", help="Output JSON file for pseudo labels")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="Root directory for training data")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of training epochs")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_training_and_inference(
        test_dir=args.test_dir,
        output_json=args.output_json,
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        resume_checkpoint=args.resume_checkpoint
    )
