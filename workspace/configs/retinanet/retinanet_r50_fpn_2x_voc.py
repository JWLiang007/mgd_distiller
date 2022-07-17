_base_ = './retinanet_r50_fpn_1x_voc.py'
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)


# =====
optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))
batch_size = 4
data = dict(
    samples_per_gpu=batch_size,
)
optimizer = dict( lr=0.01 / (16/batch_size))