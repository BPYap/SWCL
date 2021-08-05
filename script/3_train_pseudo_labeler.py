import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from retinal_fundus_encoder.config import get_argument_parser, parse_and_process_arguments
from retinal_fundus_encoder.dataset import OIAODIRDataset, EyePACSDataset
from retinal_fundus_encoder.metrics import ClassificationMetrics
from retinal_fundus_encoder.model import ClassificationModel, ClassificationModelForCAM
from retinal_fundus_encoder.train_utils import Callback, JobRunner
from retinal_fundus_encoder.transforms import TransformsDataset, EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD
from retinal_fundus_encoder.triplet_loss import TripletLoss

TRANSFORM_TRAIN_CLS = transforms.Compose([
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize(EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD)
])

TRANSFORM_EVAL_CLS = transforms.Compose([
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(EYEPACS_DATASET_MEAN, EYEPACS_DATASET_STD)
])


class S4LCallback(Callback):
    def __init__(self, unlabeled_dataloader):
        self.classification_criterion = None
        self.triplet_loss = None
        self.metrics = None

        self.unlabeled_dataloader = unlabeled_dataloader
        self.unlabeled_iterator = iter(unlabeled_dataloader)

    def init_training_criteria(self):
        self.classification_criterion = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=0.5)

    def compute_training_loss(self, args, model, labeled_batch, train_info):
        # labeled inputs
        labeled_ins = torch.cat(labeled_batch[0]).to(args.device)
        labeled_gts = torch.cat(labeled_batch[1]['normal-abnormal']).to(args.device)

        # unlabeled inputs
        try:
            unlabeled_batch = next(self.unlabeled_iterator)
        except StopIteration:
            self.unlabeled_iterator = iter(self.unlabeled_dataloader)
            unlabeled_batch = next(self.unlabeled_iterator)
        unlabeled_ins = torch.cat(unlabeled_batch[0]).to(args.device)

        # compute logits
        feat_maps, logits = model(torch.cat([labeled_ins, unlabeled_ins]))
        labeled_logits = logits[:len(labeled_ins), :]

        # calculate supervised loss for labeled data
        labeled_loss = self.classification_criterion(labeled_logits, labeled_gts)

        # calculate self-supervised loss for both labeled and unlabeled data
        embeddings = F.normalize(F.adaptive_avg_pool2d(feat_maps, 1).flatten(1))
        lbs = len(labeled_batch[0][0])  # batch size of labeled dataset
        lnv = len(labeled_batch[0])  # number of views of labeled dataset
        triplet_loss_labels = [i for i in range(lbs)] * lnv
        ubs = len(unlabeled_batch[0][0])  # batch size of unlabeled dataset
        unv = len(unlabeled_batch[0])  # number of views of unlabeled dataset
        triplet_loss_labels += [i for i in range(lbs, lbs + ubs)] * unv
        triplet_loss_labels = torch.tensor(triplet_loss_labels).to(args.device)
        unlabeled_loss = self.triplet_loss(embeddings, triplet_loss_labels)

        return labeled_loss + unlabeled_loss

    def init_metrics(self):
        self.metrics = ClassificationMetrics(2)

    def update_metrics(self, args, model, batch):
        inputs = torch.cat(batch[0]).to(args.device)
        ground_truths = torch.cat(batch[1]['normal-abnormal']).to(args.device)

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
    parser = get_argument_parser()

    # Add custom arguments
    parser.add_argument(
        "--odir_images",
        type=str,
        help="Directory consisting of training images for the OIA-ODIR dataset."
    )
    parser.add_argument(
        "--odir_labels",
        type=str,
        help="Path to training labels (.csv file) for the OIA-ODIR dataset."
    )
    parser.add_argument(
        "--eyepacs_images",
        type=str,
        help="Directory consisting of unlabeled training images for the EyePACS dataset."
    )
    parser.add_argument(
        "--eval_images",
        type=str,
        help="Directory consisting of evaluation images for the OIA-ODIR dataset."
    )
    parser.add_argument(
        "--eval_labels",
        type=str,
        help="Path to evaluation labels (.csv file) for the OIA-ODIR dataset."
    )
    parser.add_argument(
        "--reduce_stride",
        action='store_true',
        help="set the stride of last downsampling layer to 1."
    )

    args = parse_and_process_arguments(parser)

    model_constructor = ClassificationModelForCAM if args.reduce_stride else ClassificationModel
    model = model_constructor(n_channels=3, num_classes=2)

    train_dataset = OIAODIRDataset(args.odir_images, args.odir_labels, num_views=4)
    eval_dataset = None
    if args.eval_images and args.eval_labels:
        eval_dataset = OIAODIRDataset(args.eval_images, args.eval_labels)

    unlabeled_dataset = EyePACSDataset(args.eyepacs_images, num_views=4)
    unlabeled_dataloader = torch.utils.data.DataLoader(
        TransformsDataset(unlabeled_dataset, TRANSFORM_TRAIN_CLS),
        batch_size=args.per_device_train_batch_size * max(1, args.n_gpu),
        sampler=torch.utils.data.RandomSampler(unlabeled_dataset),
        num_workers=args.num_workers,
        pin_memory=False
    )

    callback = S4LCallback(unlabeled_dataloader)
    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_CLS,
        eval_transforms_image=TRANSFORM_EVAL_CLS
    )


if __name__ == "__main__":
    main()
