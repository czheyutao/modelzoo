from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法

custom_imports = dict(
    imports=['mmpretrain.models.retrievers.image2image'],
    allow_failed_imports=False
)

print("Config file loaded, custom_imports =", custom_imports)

# 你后续配置...

import mmpretrain.models.retrievers.image2image

auto_scale_lr = dict(base_batch_size=256, enable=True)
custom_hooks = [
    dict(type='PrepareProtoBeforeValLoopHook'),
    dict(type='SyncBuffersHook'),
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=3997,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'InShop'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='tccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
gallery_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/data/suda-data/css/arcface/data/inshop/Img',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='gallery',
        type='InShop'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
launcher = 'none'
load_from = None
log_level = 'INFO'

model = dict(
    head=dict(
        in_channels=2048,
        init_cfg=None,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=3997,
        type='ArcFaceClsHead'),
    image_encoder=[
        dict(
            depth=50,
            init_cfg=dict(
                checkpoint=
                '/data/suda-data/css/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth',
                prefix='backbone',
                type='Pretrained'),
            type='ResNet'),
        dict(type='GlobalAveragePooling'),
    ],
    prototype=dict(
        batch_size=32,
        dataset=dict(
            data_root='/data/suda-data/css/arcface/data/inshop/Img',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(scale=512, type='Resize'),
                dict(crop_size=448, type='CenterCrop'),
                dict(type='PackInputs'),
            ],
            split='gallery',
            type='InShop'),
        num_workers=4,
        sampler=dict(shuffle=False, type='DefaultSampler')),
    type='ImageToImageRetriever')


optim_wrapper = dict(
    optimizer=dict(
        lr=0.02, momentum=0.9, nesterov=True, type='SGD', weight_decay=0.0005))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        by_epoch=True,
        begin=5,
        end=50)
]


pretrained = '/data/suda-data/css/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'
query_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/data/suda-data/css/arcface/data/inshop/Img',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='query',
        type='InShop'),
    num_workers=4,
    sampler=dict(shuffle=False, type='DefaultSampler'))
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/data/suda-data/css/arcface/data/inshop/Img',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='query',
        type='InShop'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(topk=1, type='RetrievalRecall'),
    dict(topk=10, type='RetrievalAveragePrecision'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=512, type='Resize'),
    dict(crop_size=448, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/data/suda-data/css/arcface/data/inshop/Img',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='RandomCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='InShop'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=512, type='Resize'),
    dict(crop_size=448, type='RandomCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/data/suda-data/css/arcface/data/inshop/Img',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=512, type='Resize'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='query',
        type='InShop'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(topk=1, type='RetrievalRecall'),
    dict(topk=10, type='RetrievalAveragePrecision'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Visualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs'
