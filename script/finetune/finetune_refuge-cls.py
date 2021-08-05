import torch
import torch.nn as nn

from retinal_fundus_encoder.config import get_argument_parser, parse_and_process_arguments
from retinal_fundus_encoder.dataset import REFUGEGradingDataset
from retinal_fundus_encoder.metrics import ClassificationMetrics
from retinal_fundus_encoder.model import ClassificationModel
from retinal_fundus_encoder.train_utils import Callback, JobRunner
from retinal_fundus_encoder.transforms import TRANSFORM_TRAIN_CLS, TRANSFORM_EVAL_CLS


class REFUGEGradingCallback(Callback):
    def __init__(self):
        self.classification_criterion = None
        self.metrics = None

    def init_training_criteria(self):
        self.classification_criterion = nn.CrossEntropyLoss()

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = batch[0].to(args.device)
        ground_truths = batch[1].to(args.device)

        _, logits = model(inputs)
        loss = self.classification_criterion(logits, ground_truths)

        return loss

    def init_metrics(self):
        self.metrics = ClassificationMetrics(2)

    def update_metrics(self, args, model, batch):
        inputs = batch[0].to(args.device)
        ground_truths = batch[1].to(args.device)

        _, logits = model(inputs)
        probs = torch.softmax(logits, 1)

        self.metrics.update(probs, ground_truths)

    def summarize_metrics(self):
        summary = {}
        metrics = self.metrics

        summary.update({
            "accuracies": metrics.accuracies,
            "precisions": metrics.precisions,
            "recalls": metrics.recalls,
            "specificities": metrics.specificities,
            "F1 scores": metrics.f1_scores,
            "overall accuracy": metrics.overall_accuracy,
            "roc-auc": metrics.roc_auc_scores[1],
            "precision": metrics.precisions[1],
            "recall": metrics.recalls[1],
            "F1 score": metrics.f1_scores[1],
        })

        return summary


def main():
    parser = get_argument_parser(finetune=True)

    # Add custom arguments
    parser.add_argument(
        "--train_image_dir",
        type=str,
        help="Directory consisting of training images."
    )
    parser.add_argument(
        "--train_label_path",
        type=str,
        help="Path to training labels (.csv file)."
    )
    parser.add_argument(
        "--eval_image_dir",
        type=str,
        help="Directory consisting of evaluation images."
    )
    parser.add_argument(
        "--eval_label_path",
        type=str,
        help="Path to evaluation labels (.csv file)."
    )

    args = parse_and_process_arguments(parser)

    callback = REFUGEGradingCallback()
    model = ClassificationModel(n_channels=3, num_classes=2)
    train_dataset = None
    if args.train_image_dir and args.train_label_path:
        train_dataset = REFUGEGradingDataset(args.train_image_dir, args.train_label_path)
    eval_dataset = None
    if args.eval_image_dir and args.eval_label_path:
        eval_dataset = REFUGEGradingDataset(args.eval_image_dir, args.eval_label_path)

    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_CLS,
        eval_transforms_image=TRANSFORM_EVAL_CLS
    )


if __name__ == "__main__":
    main()
