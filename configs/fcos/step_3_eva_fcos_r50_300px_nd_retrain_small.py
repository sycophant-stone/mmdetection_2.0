# model settings
model = dict(
    type='FCOS',
    pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 16, 16, 32],
        # strides=[8, 16, 32, 64, 128],
        regress_ranges=((-1, 50), (50, 100), (100, 150), (150, 200),
                        (200, 1e8)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoHeadDataset'
data_root = '/ssd/hnren/Data/subtrain/subsets/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train_subset.json',
        img_prefix=data_root + 'JPEGImages/',
        img_scale=(300, 300),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train_subset.json',
        img_prefix=data_root + 'JPEGImages/',
        img_scale=(300, 300),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train_subset.json',
        # img_prefix=data_root + 'JPEGImages/',
        ann_file='/ssd/hnren/Data/coco_300px_clean_head_beta/annotations/instances_train.json',
        img_prefix='/ssd/hnren/Data/coco_300px_clean_head_beta/JPEGImages/',
        img_scale=(300, 300),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0,  # 0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0., norm_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='fixed',
    # policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
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
total_epochs = 30
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/step_5_allbegin_eva_r50_300px_nd'
load_from = None
resume_from = None
workflow = [('train', 1)]

# +---------------------------+---------------------------------------------------+
# |step_3_eva_r50_300px_nd    | use subsets retrain..                             |
# |                           |                                                   |
# +-------------------------------------------------------------------------------+
# |step_4_eva_r50_300px_nd    | leveldown with small regress regions..            |
# |                           |                                                   |
# |                           |                                                   |
# |                           |                                                   |
# |                           |                                                   |
# +---------------------------+---------------------------------------------------+
