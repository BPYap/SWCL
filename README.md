# SWCL
This is the official code repository for the paper "[Semi-weakly Supervised Contrastive Representation
Learning for Retinal Fundus Images](https://arxiv.org/abs/2108.02122)".

## Installation
```
python -m virtualenv env
source env/bin/activate

pip install -r requirements.txt
python setup.py install
```

## Download links - pretrained weights (PyTorch)
- [Pseudo-labeler (ResNet-34 with reduced stride)](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EoU_B0FhmhVBnIwB578kRK4BALLed7r4VkasPikJh1uSvg?e=szw9dm)
- [SWCL (ResNet-18)](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/Ev3BjboflPFHkp2bEpdeGugBLdjAwcGtHA1J1vyBrW_rWg?e=wXB9Lb)
- [SWCL (ResNetv2-50x1)](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EpE_suev4MJFq4ZIkaFPpwABOtrembgLLr7UKW6o7C4EzQ?e=y43l1Q)

## Download links - preprocessed datasets
- [Kaggle-EyePACS](https://entuedu-my.sharepoint.com/:u:/g/personal/boonpeng001_e_ntu_edu_sg/EX7LeAu8GhRCk_CkcUBJ_moBCDi1bAC8feDih0yUY8tB1A?e=JJfgz6)
- [OIA-ODIR](https://entuedu-my.sharepoint.com/:u:/g/personal/boonpeng001_e_ntu_edu_sg/EcpeG6S5FEtImvXg3OUAj_MBTIre45Lkt47B_UbjfTf5Ag?e=XNwlaA) 
- [retinal-SWAP](https://entuedu-my.sharepoint.com/:u:/g/personal/boonpeng001_e_ntu_edu_sg/EVHAHVJM5VpGp-JBv6EtWswBl7glH-B_ybsJfjUZQhPkHw?e=wBC24I)

## Reproducibility guide
##### Step 1: Download and preprocess the Kaggle-EyePACS dataset
Original dataset can be obtained from
- https://www.kaggle.com/c/diabetic-retinopathy-detection/data (for images & train labels),
- https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/16149 (for test labels).
```
# resize images
python script/0_resize_image.py --image_folder data/download/EyePACS/train --output_folder data/download/EyePACS/train_resized --size 448
python script/0_resize_image.py --image_folder data/download/EyePACS/test --output_folder data/download/EyePACS/test_resized --size 448

# merge train and test datasets
python script/1_preprocess_eyepacs.py \
    --train_images data/download/EyePACS/train_resized \
    --train_labels data/download/EyePACS/trainLabels.csv \
    --test_images data/download/EyePACS/test_resized \
    --test_labels data/download/EyePACS/retinopathy_solution.csv \
    --output data/pretrain/kaggle-eyepacs
```

##### Step 2: Download and preprocess the OIA-ODIR dataset
Original dataset can be obtained from https://github.com/nkicsl/OIA-ODIR.
```
# resize images
python script/0_resize_image.py --image_folder "data/download/OIA-ODIR/Training Set/Images" --output_folder "data/download/OIA-ODIR/Training Set/Images_resized" --size 448
python script/0_resize_image.py --image_folder "data/download/OIA-ODIR/Off-site Test Set/Images" --output_folder "data/download/OIA-ODIR/Off-site Test Set/Images_resized" --size 448
python script/0_resize_image.py --image_folder "data/download/OIA-ODIR/On-site Test Set/Images" --output_folder "data/download/OIA-ODIR/On-site Test Set/Images_resized" --size 448

# merge datasets and extract image-level labels
python script/2_preprocess_odir.py \
    --train_images "data/download/OIA-ODIR/Training Set/Images_resized" \
    --train_labels "data/download/OIA-ODIR/Training Set/Annotation/training annotation (English).xlsx" \
    --dev_images "data/download/OIA-ODIR/Off-site Test Set/Images_resized" \
    --dev_labels "data/download/OIA-ODIR/Off-site Test Set/Annotation/off-site test annotation (English).xlsx" \
    --test_images "data/download/OIA-ODIR/On-site Test Set/Images_resized" \
    --test_labels "data/download/OIA-ODIR/On-site Test Set/Annotation/on-site test annotation (English).xlsx" \
    --output_folder data/pretrain/oia-odir
```

##### Step 3: Train a pseudo-labeler with S<sup>4</sup>L
```
python script/3_train_pseudo_labeler.py \
    --model_dir model/pseudo-labeler_resnet-34-rs \
    --n_gpu 8 \
    --num_workers 5 \
    --encoder_backbone resnet-34 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 200 \
    --lr_scheduler step \
    --step_decay_milestones 140,160,180 \
    --step_decay_factor 0.1 \
    --learning_rate 0.03 \
    --optimizer sgd \
    --momentum 0.9 \
    --nesterov \
    --weight_decay 0.001 \
    --logging_steps 280 \
    --save_steps 280 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --odir_images data/pretrain/oia-odir/train \
    --odir_labels data/pretrain/oia-odir/train.csv \
    --eyepacs_images data/pretrain/kaggle-eyepacs/images \
    --eval_images data/pretrain/oia-odir/dev \
    --eval_labels data/pretrain/oia-odir/dev.csv \
    --reduce_stride
```

##### Step 4: Generate the semi-weakly annoated patch dataset (retinal-SWAP)
```
python script/4_generate_patch_dataset.py \
    --eyepacs_images data/pretrain/kaggle-eyepacs/images \
    --oiaodir_images data/pretrain/oia-odir/dev \
    --oiaodir_labels data/pretrain/oia-odir/dev.csv \
    --model_path model/pseudo-labeler_resnet-34-rs/pytorch_model.bin \
    --encoder_backbone resnet-34 \
    --output_folder data/pretrain/retinal-swap \
    --crop_size 224
```

##### Step 5: Pretrain an encoder-projector network with SWCL
```
python script/5_pretrain_swcl.py \
    --model_dir model/swcl_resnetv2-50x1 \
    --n_gpu 8 \
    --encoder_backbone resnetv2-50x1 \
    --num_workers 16 \
    --per_device_train_batch_size 31 \
    --num_train_epochs 40 \
    --optimizer adam \
    --warmup_epochs 5 \
    --learning_rate 0.001 \
    --weight_decay 1e-4 \
    --logging_steps 995 \
    --save_steps 19900 \
    --do_train \
    --image_dir data/pretrain/retinal-swap/patches \
    --label_path data/pretrain/retinal-swap/labels.csv \
    --label_scheme all \
    --abnormality_threshold 0.4 \
    --temperature 0.1
```

##### Step 6: Fine-tune on downstream tasks
The resized downstream datasets used in the paper are available for download [here](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/Ei4JmTZjDL1BseeJxE6OKRgBKRA8WXV4Do7F4hf9ZV7U7g?e=X6ns0C).

Example for glaucoma classification on REFUGE-cls dataset:
```
# train
python script/finetune/finetune_refuge-cls.py \
    --model_dir "model/finetune/refuge-cls/swcl" \
    --train_image_dir "data/finetune/REFUGE/train/images-resized" \
    --train_label_path "data/finetune/REFUGE/train/labels.csv" \
    --eval_image_dir "data/finetune/REFUGE/val/images-resized" \
    --eval_label_path "data/finetune/REFUGE/val/labels.csv" \
    --n_gpu 1 \
    --num_workers 1 \
    --per_device_eval_batch_size 16 \
    --logging_steps 30 \
    --evaluate_during_training \
    --do_train \
    --momentum 0.9 \
    --nesterov \
    --encoder_backbone resnetv2-50x1 \
    --per_device_train_batch_size 64 \
    --num_train_epochs 120 \
    --lr_scheduler "none" \
    --pretrain_model_path model/swcl_resnetv2-50x1/pytorch_model.bin \
    --learning_rate 0.1 \
    --weight_decay 0.001
    
# eval
python script/finetune/finetune_refuge-cls.py \
    --model_dir "model/finetune/refuge-cls/swcl" \
    --eval_image_dir "data/finetune/REFUGE/test/images-resized" \
    --eval_label_path "data/finetune/REFUGE/test/labels.csv" \
    --n_gpu 1 \
    --num_workers 1 \
    --per_device_eval_batch_size 16 \
    --do_eval \
    --encoder_backbone resnetv2-50x1
```

Example for joint optic disc and optic cup segmentation on REFUGE-seg dataset:
```
# train
python script/finetune/finetune_refuge-seg.py \
    --model_dir "model/finetune/refuge-seg/swcl" \
    --train_image_dir "data/finetune/REFUGE/train/images-resized-seg" \
    --train_mask_dir "data/finetune/REFUGE/train/segmentations-resized" \
    --eval_image_dir "data/finetune/REFUGE/val/images-resized-seg" \
    --eval_mask_dir "data/finetune/REFUGE/val/segmentations-resized" \
    --n_gpu 1 \
    --num_workers 1 \
    --per_device_eval_batch_size 16 \
    --logging_steps 500 \
    --evaluate_during_training \
    --do_train \
    --momentum 0.9 \
    --nesterov \
    --encoder_backbone resnetv2-50x1 \
    --segmentation_architecture deeplabv3+ \
    --per_device_train_batch_size 8 \
    --num_train_epochs 250 \
    --lr_scheduler "none" \
    --pretrain_model_path model/swcl_resnetv2-50x1/pytorch_model.bin \
    --learning_rate 0.1 \
    --weight_decay 0.0005

# eval
python script/finetune/finetune_refuge-seg.py \
    --model_dir "model/finetune/refuge-seg/swcl" \
    --eval_image_dir "data/finetune/REFUGE/test/images-resized-seg" \
    --eval_mask_dir "data/finetune/REFUGE/test/segmentations" \
    --n_gpu 1 \
    --num_workers 1 \
    --per_device_eval_batch_size 16 \
    --do_eval \
    --encoder_backbone resnetv2-50x1 \
    --segmentation_architecture deeplabv3+
```

## Citation
```
@misc{yap2021semiweakly,
      title={Semi-weakly Supervised Contrastive Representation Learning for Retinal Fundus Images}, 
      author={Boon Peng Yap and Beng Koon Ng},
      year={2021},
      eprint={2108.02122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```