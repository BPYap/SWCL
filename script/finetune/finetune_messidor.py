import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from retinal_fundus_encoder.config import get_argument_parser, parse_and_process_arguments
from retinal_fundus_encoder.dataset import MessidorDataset
from retinal_fundus_encoder.metrics import ClassificationMetrics
from retinal_fundus_encoder.model import JointClassificationModel
from retinal_fundus_encoder.train_utils import Callback, JobRunner
from retinal_fundus_encoder.transforms import TRANSFORM_TRAIN_CLS, TRANSFORM_EVAL_CLS

DR_TASK_NAME = 'retinopathy-referability'
ME_TASK_NAME = 'macular-edema-risk'
NUM_CLASS_PER_TASK = {
    DR_TASK_NAME: len(MessidorDataset.TASK_LABELS[DR_TASK_NAME]),
    ME_TASK_NAME: len(MessidorDataset.TASK_LABELS[ME_TASK_NAME])
}


class MessidorCallback(Callback):
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
            is_binary = metrics.num_classes == 2
            summary.update({
                f"{prefix} accuracies": metrics.accuracies,
                f"{prefix} precisions": metrics.precisions,
                f"{prefix} recalls": metrics.recalls,
                f"{prefix} specificities": metrics.specificities,
                f"{prefix} F1 scores": metrics.f1_scores,
                f"{prefix} overall accuracy": metrics.overall_accuracy,
                f"{prefix} roc-auc": metrics.macro_roc_auc,
                f"{prefix} precision": metrics.precisions[1] if is_binary else metrics.macro_precision,
                f"{prefix} recall": metrics.recalls[1] if is_binary else metrics.macro_recall,
                f"{prefix} F1 score": metrics.f1_scores[1] if is_binary else metrics.macro_f1,
            })

        bitmap = ((torch.stack(dr_metrics.predicted_probs).argmax(dim=0) == dr_metrics.ground_truths) &
                  (torch.stack(me_metrics.predicted_probs).argmax(dim=0) == me_metrics.ground_truths))
        summary["joint accuracy"] = bitmap.sum().item() / len(dr_metrics.ground_truths)

        return summary


def main():
    parser = get_argument_parser(finetune=True)

    # Add custom arguments
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory consisting of images grouped by ophthalmology department."
    )
    parser.add_argument(
        "--cross_validation_folds",
        type=int,
        default=-1,
        help="Specify number of folds to run cross-validation. Overwrite `--do_train`, `--do_eval` and "
             "`--random_validation`."
    )

    args = parse_and_process_arguments(parser)

    full_dataset = MessidorDataset(args.dataset_dir)

    if args.cross_validation_folds > 1:
        folds = StratifiedKFold(n_splits=args.cross_validation_folds)
        labels = [row['Retinopathy grade'] for _, row in full_dataset.data.iterrows()]
        for train_indices, test_indices in folds.split(np.zeros(len(full_dataset)), labels):
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            eval_dataset = torch.utils.data.Subset(full_dataset, test_indices)

            args.do_train = True
            args.do_eval = False
            args.random_validation = False
            callback = MessidorCallback()
            model = JointClassificationModel(n_channels=3, num_class_per_task=NUM_CLASS_PER_TASK)
            job_runner = JobRunner(args, model, callback)
            job_runner.run(
                train_dataset, eval_dataset=eval_dataset,
                train_transforms_image=TRANSFORM_TRAIN_CLS,
                eval_transforms_image=TRANSFORM_EVAL_CLS
            )

            args.do_train = False
            args.do_eval = True
            callback = MessidorCallback()
            model = JointClassificationModel(n_channels=3, num_class_per_task=NUM_CLASS_PER_TASK)
            job_runner = JobRunner(args, model, callback)
            job_runner.run(
                train_dataset=None, eval_dataset=eval_dataset,
                eval_transforms_image=TRANSFORM_EVAL_CLS
            )
    else:
        callback = MessidorCallback()
        model = JointClassificationModel(n_channels=3, num_class_per_task=NUM_CLASS_PER_TASK)
        job_runner = JobRunner(args, model, callback)
        job_runner.run(
            full_dataset, eval_dataset=None,
            train_transforms_image=TRANSFORM_TRAIN_CLS,
            eval_transforms_image=TRANSFORM_EVAL_CLS
        )


if __name__ == "__main__":
    main()
