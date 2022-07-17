_base_ = [
    '../../retinanet/retinanet_r50_fpn_2x_voc.py'
]
# model settings
find_unused_parameters=True
alpha_mgd=0.00002
lambda_mgd=0.65
alpha_adv=0.00002
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = 'result/retinanet/retinanet_x101_voc_24.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       ),
                                dict(type='AdvLoss',
                                       name='adv_loss_mgd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_adv=alpha_adv,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       ),
                                dict(type='AdvLoss',
                                       name='adv_loss_mgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_adv=alpha_adv,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       ),
                                  dict(type='AdvLoss',
                                       name='adv_loss_mgd_fpn_2',
                                       student_channels=256,
                                       teacher_channels=256,
                                       alpha_adv=alpha_adv,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       ),
                                  dict(type='AdvLoss',
                                       name='adv_loss_mgd_fpn_1',
                                       student_channels=256,
                                       teacher_channels=256,
                                       alpha_adv=alpha_adv,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       ),
                                  dict(type='AdvLoss',
                                       name='adv_loss_mgd_fpn_0',
                                       student_channels=256,
                                       teacher_channels=256,
                                       alpha_adv=alpha_adv,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/retinanet/retinanet_r50_fpn_2x_voc.py'
teacher_cfg = 'configs/retinanet/retinanet_x101_64x4d_fpn_1x_voc.py'

# ===================
batch_size = 4
optimizer = dict( lr=0.01 / (16/batch_size))
optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))

train_pipeline = [
    dict(type='LoadImageFromFile',adv_img='/home/jwl/code/mga_compare_ss/data/rtn/exp/5/'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','adv' ,'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        pipeline=train_pipeline),
)