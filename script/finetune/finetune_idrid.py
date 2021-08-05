import torch
import torch.nn as nn

from retinal_fundus_encoder.config import get_argument_parser, parse_and_process_arguments
from retinal_fundus_encoder.dataset import IDRiDGradingDataset
from retinal_fundus_encoder.metrics import ClassificationMetrics
from retinal_fundus_encoder.model import JointClassificationModel
from retinal_fundus_encoder.train_utils import Callback, JobRunner
from retinal_fundus_encoder.transforms import TRANSFORM_TRAIN_CLS, TRANSFORM_EVAL_CLS

DR_TASK_NAME = 'retinopathy-grade'
ME_TASK_NAME = 'macular-edema-risk'
NUM_CLASS_PER_TASK = {
    DR_TASK_NAME: len(IDRiDGradingDataset.TASK_LABELS[DR_TASK_NAME]),
    ME_TASK_NAME: len(IDRiDGradingDataset.TASK_LABELS[ME_TASK_NAME])
}


class IDRiDGradingCallback(Callback):
    def __init__(self):
        self.task_names = [DR_TASK_NAME, ME_TASK_NAME]

        self.classification_criterion = None
        self.per_task_metrics = None

    def init_training_criteria(self):
        self.classification_criterion = nn.CrossEntropyLoss()

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = batch[0].to(args.device)
        _, logits = model(inputs)
        loss = 0.0
        for task_name in self.task_names:
            ground_truths = batch[1][task_name].to(args.device)
            loss += self.classification_criterion(logits[task_name], ground_truths)

        return loss

    def init_metrics(self):
        self.per_task_metrics = {t: ClassificationMetrics(NUM_CLASS_PER_TASK[t]) for t in self.task_names}

    def update_metrics(self, args, model, batch):
        inputs = batch[0].to(args.device)
        _, logits = model(inputs)
        for task_name in self.task_names:
            probs = torch.softmax(logits[task_name], 1)
            ground_truths = batch[1][task_name].to(args.device)

            self.per_task_metrics[task_name].update(probs, ground_truths)

    def summarize_metrics(self):
        summary = {}
        dr_metrics = self.per_task_metrics[DR_TASK_NAME]
        me_metrics = self.per_task_metrics[ME_TASK_NAME]
        for task_name, metrics in zip(self.task_names, [dr_metrics, me_metrics]):
            prefix = f"[{task_name}]"
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

        bitmap = ((torch.stack(dr_metrics.predicted_probs).argmax(dim=0) == dr_metrics.ground_truths) &
                  (torch.stack(me_metrics.predicted_probs).argmax(dim=0) == me_metrics.ground_truths))
        summary["joint accuracy"] = bitmap.sum().item() / len(dr_metrics.ground_truths)

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

    callback = IDRiDGradingCallback()
    model = JointClassificationModel(n_channels=3, num_class_per_task=NUM_CLASS_PER_TASK)
    train_dataset = None
    if args.train_image_dir and args.train_label_path:
        train_dataset = IDRiDGradingDataset(args.train_image_dir, args.train_label_path)
    eval_dataset = None
    if args.eval_image_dir and args.eval_label_path:
        eval_dataset = IDRiDGradingDataset(args.eval_image_dir, args.eval_label_path)

    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_CLS,
        eval_transforms_image=TRANSFORM_EVAL_CLS
    )


if __name__ == "__main__":
    main()
