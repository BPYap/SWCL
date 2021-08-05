import torch
import torch.nn as nn

from retinal_fundus_encoder.config import get_argument_parser, parse_and_process_arguments
from retinal_fundus_encoder.dataset import CHASEDB1Dataset
from retinal_fundus_encoder.metrics import SegmentationMetrics
from retinal_fundus_encoder.model import SegmentationModel
from retinal_fundus_encoder.script_utils import forward_patches
from retinal_fundus_encoder.train_utils import Callback, JobRunner
from retinal_fundus_encoder.transforms import (TRANSFORM_TRAIN_SEG, TRANSFORM_TRAIN_SEG_MASK,
                                               TRANSFORM_EVAL_SEG, TRANSFORM_EVAL_SEG_MASK)


class CHASEDB1Callback(Callback):
    def __init__(self):
        self.segmentation_criterion = None
        self.metrics = None

    def init_training_criteria(self):
        self.segmentation_criterion = nn.CrossEntropyLoss()

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = batch[0].to(args.device)
        ground_truths = batch[1].to(args.device)
        logits = model(inputs)

        return self.segmentation_criterion(logits, ground_truths)

    def init_metrics(self):
        self.metrics = SegmentationMetrics(2)

    def update_metrics(self, args, model, batch):
        inputs = batch[0].to(args.device)
        ground_truths = batch[1].to(args.device)
        logits = forward_patches(model, inputs)
        predictions = torch.argmax(logits, dim=1)

        self.metrics.update(predictions, ground_truths)

    def summarize_metrics(self):
        metrics = self.metrics

        return {
            "F1 score": metrics.f1_scores[1],
        }


def main():
    parser = get_argument_parser(segmentation=True, finetune=True)

    # Add custom arguments
    parser.add_argument(
        "--train_image_dir",
        type=str,
        help="Directory consisting of training images."
    )
    parser.add_argument(
        "--train_mask_dir",
        type=str,
        help="Directory consisting of training segmentation masks."
    )
    parser.add_argument(
        "--eval_image_dir",
        type=str,
        help="Directory consisting of evaluation images."
    )
    parser.add_argument(
        "--eval_mask_dir",
        type=str,
        help="Directory consisting of evaluation segmentation masks."
    )

    args = parse_and_process_arguments(parser)

    callback = CHASEDB1Callback()
    model = SegmentationModel(n_channels=3, num_classes=2)
    train_dataset = None
    if args.train_image_dir and args.train_mask_dir:
        train_dataset = CHASEDB1Dataset(args.train_image_dir, args.train_mask_dir)
    eval_dataset = None
    if args.eval_image_dir and args.eval_mask_dir:
        eval_dataset = CHASEDB1Dataset(args.eval_image_dir, args.eval_mask_dir)

    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_SEG, train_transforms_mask=TRANSFORM_TRAIN_SEG_MASK,
        eval_transforms_image=TRANSFORM_EVAL_SEG, eval_transforms_mask=TRANSFORM_EVAL_SEG_MASK
    )


if __name__ == "__main__":
    main()
