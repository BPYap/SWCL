import torch
import torch.nn as nn

from retinal_fundus_encoder.config import get_argument_parser, parse_and_process_arguments
from retinal_fundus_encoder.dataset import PreTrainMultitaskDataset
from retinal_fundus_encoder.model import JointClassificationModel
from retinal_fundus_encoder.train_utils import Callback, JobRunner
from retinal_fundus_encoder.transforms import TRANSFORM_TRAIN_CLS, TRANSFORM_EVAL_CLS
from retinal_fundus_encoder.metrics import ClassificationMetrics

ABNORMALITY_TASK_NAME = 'normal-abnormal'
POSITION_TASK_NAME = 'position'
NUM_CLASS_PER_TASK = {
    ABNORMALITY_TASK_NAME: 2,
    POSITION_TASK_NAME: 5
}

TASK_NAMES = [ABNORMALITY_TASK_NAME, POSITION_TASK_NAME]
LOSS_WEIGHTS = {
    ABNORMALITY_TASK_NAME: 1,
    POSITION_TASK_NAME: 0.1
}


class PretrainCallback(Callback):
    def __init__(self):
        self.task_names = TASK_NAMES

        self.classification_criterion = None
        self.per_task_metrics = None

    def init_training_criteria(self):
        self.classification_criterion = nn.CrossEntropyLoss()

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = torch.cat(batch[0]).to(args.device)
        _, logits = model(inputs)
        loss = 0.0
        for task_name in self.task_names:
            ground_truths = torch.cat(batch[1][task_name]).to(args.device)
            loss += LOSS_WEIGHTS[task_name] * self.classification_criterion(logits[task_name], ground_truths)

        return loss

    def init_metrics(self):
        self.per_task_metrics = {t: ClassificationMetrics(NUM_CLASS_PER_TASK[t]) for t in self.task_names}

    def update_metrics(self, args, model, batch):
        inputs = torch.cat(batch[0]).to(args.device)
        _, logits = model(inputs)
        for task_name in self.task_names:
            probs = torch.softmax(logits[task_name], 1)
            ground_truths = torch.cat(batch[1][task_name]).to(args.device)

            self.per_task_metrics[task_name].update(probs, ground_truths)

    def summarize_metrics(self):
        summary = {}
        for task_name in self.task_names:
            prefix = f"[{task_name}]"
            metrics = self.per_task_metrics[task_name]
            summary.update({
                f"{prefix} accuracies": metrics.accuracies,
                f"{prefix} precisions": metrics.precisions,
                f"{prefix} recalls": metrics.recalls,
                f"{prefix} specificities": metrics.specificities,
                f"{prefix} F1 scores": metrics.f1_scores,
                f"{prefix} overall accuracy": metrics.overall_accuracy,
                f"{prefix} auc": metrics.macro_roc_auc,
                f"{prefix} precision": metrics.macro_precision,
                f"{prefix} recall": metrics.macro_recall,
                f"{prefix} F1 score": metrics.macro_f1,
            })

        return summary


def main():
    parser = get_argument_parser(distributed=True)

    # Add custom arguments
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory consisting of training images generated from `preprocess_contrastive.py`."
    )
    parser.add_argument(
        "--label_path",
        type=str,
        required=True,
        help="Path to label file generated from `preprocess_contrastive.py`."
    )
    parser.add_argument(
        "--abnormality_threshold",
        type=float,
        default=0.5,
        help="Threshold for abnormality scores to be considered as positives."
    )

    args = parse_and_process_arguments(parser)

    callback = PretrainCallback()
    model = JointClassificationModel(n_channels=3, num_class_per_task={t: NUM_CLASS_PER_TASK[t] for t in TASK_NAMES})
    train_dataset = PreTrainMultitaskDataset(args.image_dir, args.label_path, args.abnormality_threshold)

    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset=None,
        train_transforms_image=TRANSFORM_TRAIN_CLS,
        eval_transforms_image=TRANSFORM_EVAL_CLS
    )


if __name__ == "__main__":
    main()
