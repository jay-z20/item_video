# model settings
import os

pretrained = '/mmdetection/pretrained/resnet101_ibn_a.pth.tar'
backbone = dict(
    type='resnet101_ibn_a',
    last_stride=1)

bbox_roi_extractor = dict(
    type='SingleRoIExtractor',
    roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
    out_channels=2048,
    featmap_strides=[16])

loss = dict(
    METRIC_LOSS_TYPE = 'triplet_center',
    IF_LABELSMOOTH='off',
    IF_WITH_CENTER='yes',
    SOLVER=dict(
        CLUSTER_MARGIN=0.3,
        CENTER_LR=0.5,
        CENTER_LOSS_WEIGHT=0.0005,
        RANGE_K=2,
        RANGE_MARGIN=0.3,
        RANGE_ALPHA=0,
        RANGE_BETA=1,
        RANGE_LOSS_WEIGHT=1,
        BIAS_LR_FACTOR=1,
        WEIGHT_DECAY=0.0005,
        WEIGHT_DECAY_BIAS=0.0005
    )
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/myspace/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True,gt_inids=True),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True,gt_inids=True),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
    #dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=(640, 360),  # 960 540
    #     flip=False,
    #     transforms=[
    #         dict(type='Resize', keep_ratio=True),
    #         dict(type='RandomFlip'),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='Pad', size_divisor=32),
    #         dict(type='ImageToTensor', keys=['img']),
    #         dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    #     ])
]
data = dict(
    ids_flag=True,
    imgs_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test_gallery_anns.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test_video=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/test_query_anns.json',
            img_prefix=data_root,
            pipeline=test_pipeline,
            test_mode=True))


# optimizer
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[1])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 1
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/myspace/work_dirs/resnet101_ibn_a'
load_from = None#'/root/code/myspace/work_dirs/resnet_x101/'  # '/mmdetection/pretrained/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth'
resume_from = None#'/myspace/work_dirs/resnet_x101/latest.pth' if os.path.exists('/myspace/work_dirs/resnet_x101/latest.pth') else None
workflow = [('train', 1)]
