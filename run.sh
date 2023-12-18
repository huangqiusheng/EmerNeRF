# !/bin/bash

# python preprocess.py \
#     --data_root data/waymo/raw/ \
#     --target_dir data/waymo/processed \
#     --split training \
#     --process_keys images lidar calib pose dynamic_masks \
#     --workers 2 \
#     --scene_ids 700

# -m debugpy --listen 0.0.0.0:27678 --wait-for-client CUDA_VISIBLE_DEVICES=0
# python -m torch.distributed.launch --nproc_per_node=3 train.py --config configs/ners_mvmc/style_lotd_neus/fmm_accel_co3d_moto.yaml --ddp 


DATE=`date '+%m%d'`

scene_idx=700
start_timestep=0
end_timestep=190
# reduce num_iters to 8000 for debugging
num_iters=25000

output_root="./delete/$DATE"
project=scene_reconstruction
python train_emernerf.py \
    --config_file configs/default_dynamic.yaml \
    --output_root $output_root \
    --project $project \
    --run_name ${scene_idx}_flow \
    --enable_wandb \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    logging.saveckpt_freq=$num_iters \
    optim.num_iters=$num_iters
    # data.pixel_source.load_features=False \
    # data.pixel_source.feature_model_type=dinov2_vitb14 \
    # nerf.model.head.enable_feature_head=True \
    # nerf.model.head.enable_learnable_pe=True \