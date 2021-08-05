import argparse
import csv
import os

import torch
from torchvision import transforms
from tqdm import tqdm

from retinal_fundus_encoder.dataset import OIAODIRDataset, EyePACSDataset
from retinal_fundus_encoder.model import ClassificationModelForCAM
from retinal_fundus_encoder.script_utils import get_cams
from retinal_fundus_encoder.transforms import TransformsDataset, EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD

BACKGROUND_COLOR_THRESHOLD = sum([(15 / 255 - EYEPACS_DATASET_MEAN[i]) / EYEPACS_DATASET_STD[i] for i in range(3)])
MEAN = torch.tensor(EYEPACS_DATASET_MEAN).reshape(3, 1, 1)
STD = torch.tensor(EYEPACS_DATASET_STD).reshape(3, 1, 1)

TASK_NAME = 'normal-abnormal'
NUM_CLASS = 2
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--eyepacs_images", type=str, required=True)
    arg_parser.add_argument("--oiaodir_images", type=str, required=True)
    arg_parser.add_argument("--oiaodir_labels", type=str, required=True)
    arg_parser.add_argument("--model_path", type=str, required=True)
    arg_parser.add_argument("--encoder_backbone", type=str, required=True)
    arg_parser.add_argument("--output_folder", type=str, required=True)
    arg_parser.add_argument("--crop_size", type=int, default=224)
    args = arg_parser.parse_args()

    patch_folder = os.path.join(args.output_folder, "patches")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        os.makedirs(patch_folder)

    crop_height = args.crop_size
    crop_width = args.crop_size

    model = ClassificationModelForCAM(n_channels=3, num_classes=NUM_CLASS)
    model.init(args.encoder_backbone, imagenet_init=False)
    model.load_state_dict(torch.load(args.model_path), strict=True)
    model.to(torch.device("cuda"))
    model.eval()

    left_patches = dict()
    right_patches = dict()
    for name, dataset in [("eyepacs", EyePACSDataset(args.eyepacs_images)),
                          ("oia-odir", OIAODIRDataset(args.oiaodir_images, args.oiaodir_labels))]:
        dataset = TransformsDataset(
            dataset,
            image_transform=transforms.Compose([
                transforms.CenterCrop(args.crop_size * 2),
                transforms.ToTensor(),
                transforms.Normalize(EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD)
            ])
        )
        data_loader = iter(torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False))

        with torch.no_grad():
            for batch in tqdm(data_loader):
                inputs = batch[0][0].to(torch.device("cuda"))
                image_ids = batch[1]["image_id"]
                image_labels = batch[1][TASK_NAME][0]

                image_height = inputs.shape[-2]
                image_width = inputs.shape[-1]

                feature_maps, logits = model(inputs)
                positive_cams = get_cams(
                    feature_maps=feature_maps,
                    classification_layer=model.out["classification"][-1],
                    targets=[POSITIVE_CLASS] * len(inputs),
                    upsample_size=inputs.shape[-2:]
                ).squeeze(dim=1)
                negative_cams = get_cams(
                    feature_maps=feature_maps,
                    classification_layer=model.out["classification"][-1],
                    targets=[NEGATIVE_CLASS] * len(inputs),
                    upsample_size=inputs.shape[-2:]
                ).squeeze(dim=1)

                # apply softmax along output dimension
                cams = torch.cat([positive_cams.unsqueeze(1), negative_cams.unsqueeze(1)], dim=1)
                cams = torch.softmax(cams, dim=1)[:, 0, :, :]

                # apply background masks
                background_masks = inputs.sum(dim=1) > BACKGROUND_COLOR_THRESHOLD
                cams = cams * background_masks

                for suffix, (h, w) in [
                    ("tl", (0, 0)),
                    ("tr", (0, image_width - crop_width)),
                    ("bl", (image_height - crop_height, 0)),
                    ("br", (image_height - crop_height, image_width - crop_width)),
                    ("c", (int((image_height - crop_height + 1) * 0.5), int((image_width - crop_width + 1) * 0.5)))
                ]:
                    img_crops = inputs[:, :, h: h + crop_height, w: w + crop_width].cpu()
                    cam_crops = cams[:, h: h + crop_height, w: w + crop_width]
                    scores = torch.flatten(cam_crops, start_dim=1).mean(dim=1)

                    for img_id, img_label, img_crop, score in zip(image_ids, image_labels, img_crops, scores):
                        img_label = img_label.item()
                        score = score.item()
                        if img_label == NEGATIVE_CLASS:
                            # set positive score to zero if the ground truth image-level label is negative
                            score = 0

                        left_right = img_id.split('_')[-1]
                        assert left_right in ['left', 'right']
                        if left_right == 'left':
                            patch_id = f"{name}_{img_id}_{suffix}"
                            left_patches[patch_id] = score
                        else:
                            # apply horizontal flip to images from the right eye to make them roughly align
                            # to their left counterparts
                            img_crop = img_crop.flip(-1)
                            flip_suffix = {"tl": "tr", "tr": "tl", "bl": "br", "br": "bl", "c": "c"}
                            patch_id = f"{name}_{img_id}_{flip_suffix[suffix]}"
                            right_patches[patch_id] = score

                        # unnormalize and save image crop
                        img_crop = img_crop * STD + MEAN
                        transforms.functional.to_pil_image(img_crop).save(
                            os.path.join(patch_folder, f"{patch_id}.jpg")
                        )

    assert len(left_patches) == len(right_patches)
    with open(os.path.join(args.output_folder, "labels.csv"), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'left-patch', "left-score", 'right-patch', 'right-score'])
        pair_id = 0
        for left_patch_id in left_patches.keys():
            right_patch_id = left_patch_id.replace("left", "right")
            left_score = left_patches[left_patch_id]
            right_score = right_patches[right_patch_id]

            csv_writer.writerow([pair_id, left_patch_id, left_score, right_patch_id, right_score])
            pair_id += 2
