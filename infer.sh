#!/bin/bash
source /D/anaconda/bin/activate
conda activate acomp_barricade

DATA="data/barricade"

model=barricade

th=0.1
echo $model, $th
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config experiments/COCOA/pcnet_m/config_train.yaml \
    --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours" \
    --order-th $th \
    --amodal-th $th \
    --annotation "data/barricade/annotations/infer.json" \
    --image-root $DATA/infer \
    --test-num -1 \
    --output experiments/COCOA/pcnet_m_$model/amodal_results/barricade_amodalcomp_$th.json

    # --load-model "experiments/COCOA/pcnet_m_barricade/checkpoints/ckpt_iter_28000.pth.tar"\


# for th in `seq 0.2 0.1 0.9`
# do
#     echo $model, $th
#     CUDA_VISIBLE_DEVICES=0 \ 
#     python tools/test.py \
#         --config experiments/COCOA/pcnet_m/config_train_barricade.yaml \
#         --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_16000.pth.tar"\
#         --order-method "ours" \
#         --amodal-method "ours" \
#         --order-th $th \
#         --amodal-th 0.5 \
#         --annotation "data/barricade/annotations/barricade_infer.json" \
#         --image-root $DATA/infer \
#         --test-num -1 \
#         --output experiments/COCOA/pcnet_m_$model/amodal_results/barricade_amodalcomp_infer_$th.json
# done


# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_barricade.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_36000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --amodal-th 0.5 \
#     --annotation "data/barricade/annotations/barricade_test.json" \
#     --image-root $DATA/test \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/barricade_amodalcomp_val.json

# echo $model, $th
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_36000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --amodal-th 0.5 \
#     --annotation "data/barricade/annotations/barricade_test.json" \
#     --image-root $DATA/test \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/barricade_amodalcomp_val.json






# model=std_no_rgb_exponential
# th=0.95
# echo $model, $th
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th $th \
#     --amodal-th 0.75 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_mumford_shah
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_occluded_only
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_gaussian_boundary
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_std_no_rgb_gaussian_boundary_reg/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_gaussian_boundary
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_std_no_rgb_gaussian_boundary_reg/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_default_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$modal/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_exponential
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb_exponential
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_mumford_shah
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_mumford_shah_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.5 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=std_no_rgb_mumford_shah
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_std_no_rgb_mumford_shah_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.5 \
#     --annotation "data/COCOA/annotations/COCO_amodal_test2014.json" \
#     --image-root $DATA/test2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json


# model=boundary_no_rgb_mumford_shah
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_mumford_shah_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json



# model=std_no_rgb_mumford_shah_mse
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=std_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=std_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json



# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_train2014.json" \
#     --image-root $DATA/train2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_train_ours.json

# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json



# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_test2014.json" \
#     --image-root $DATA/test2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json


# model=supervised_no_rgb
# # model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "sup" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=supervised_no_rgb
# # model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "sup" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_test2014.json" \
#     --image-root $DATA/test2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json


# model=supervised_no_rgb
# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/KINS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/KINS/pcnet_m_$model/checkpoints/ckpt_iter_32000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "sup" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/KINS/instances_val.json" \
#     --image-root data/KINS/testing/image_2 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=boundary_no_rgb_gaussian
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.5 \
#     --annotation "data/COCOA/annotations/COCO_amodal_test2014.json" \
#     --image-root $DATA/test2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json

# model=boundary_no_rgb_mumford_shah
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.5 \
#     --amodal-th 0.5 \
#     --annotation "data/COCOA/annotations/COCO_amodal_test2014.json" \
#     --image-root $DATA/test2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_test_ours.json



# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_test2014.json" \
#     --image-root $DATA/test2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_train_ours.json


# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/LVIS/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_boundary_no_rgb_8/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "mutual/mutual_coco.json" \
#     --image-root mutual \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/temp.json


# model=boundary_no_rgb_deeplabv3+
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json



# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json

# model=default_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "released/COCOA_pcnet_m.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=default
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_gaussian
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_exponential
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_gaussian
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb_exponential
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json


# model=boundary_no_rgb
# echo $model
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py \
#     --config experiments/COCOA/pcnet_m/config_train_$model.yaml \
#     --load-model "experiments/COCOA/pcnet_m_$model/checkpoints/ckpt_iter_56000.pth.tar"\
#     --order-method "ours" \
#     --amodal-method "ours_nog" \
#     --order-th 0.1 \
#     --amodal-th 0.2 \
#     --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
#     --image-root $DATA/val2014 \
#     --test-num -1 \
#     --output experiments/COCOA/pcnet_m_$model/amodal_results/amodalcomp_val_ours.json