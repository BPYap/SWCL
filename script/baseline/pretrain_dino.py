import math
import os

import torch

from retinal_fundus_encoder.config import get_argument_parser, parse_and_process_arguments, is_distributed
from retinal_fundus_encoder.dataset import PreTrainMultiviewDataset
from retinal_fundus_encoder.model import DINOModel
from retinal_fundus_encoder.train_utils import Callback, JobRunner
from retinal_fundus_encoder.transforms import TRANSFORM_TRAIN_CONTRASTIVE


def initialize_teacher_network(args, student_network, teacher_network):
    teacher_network.init(args.encoder_backbone, imagenet_init=False)
    teacher_network.to(args.device)
    update_teacher_network(student_network, teacher_network, momentum=0)


def update_teacher_network(student_network, teacher_network, momentum):
    if hasattr(student_network, "module"):
        student_network = student_network.module

    # encoder parameters
    for student_param, teacher_param in zip(student_network.encoder.parameters(),
                                            teacher_network.encoder.parameters()):
        teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
        teacher_param.requires_grad = False

    # projector parameters
    for student_param, teacher_param in zip(student_network.projection_network.parameters(),
                                            teacher_network.projection_network.parameters()):
        teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data
        teacher_param.requires_grad = False


def get_momentum(base_momentum, current_step, total_steps):
    return 1 - (1 - base_momentum) * (math.cos((current_step * math.pi) / total_steps) + 1) / 2


class DinoLoss(torch.nn.Module):
    def __init__(self, teacher_temp, student_temp):
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def update_teacher_temp(self, new_temp):
        self.teacher_temp = new_temp

    def forward(self, teacher_features, student_features, center_vector):
        assert teacher_features.shape == student_features.shape

        # center and sharpen teacher features
        teacher_features = torch.softmax((teacher_features - center_vector) / self.teacher_temp, dim=1)

        # sharpen student features
        student_features = torch.softmax(student_features / self.student_temp, dim=1)

        # calculate cross-entropy loss
        loss = - (teacher_features * torch.log(student_features)).sum(dim=1).mean()

        return loss


class PretrainCallback(Callback):
    def __init__(self, teacher_network, base_momentum):
        self.teacher_network = teacher_network
        self.base_momentum = base_momentum

        self.center = None
        self.distill_criterion = None

    def init_training_criteria(self):
        self.distill_criterion = DinoLoss(teacher_temp=0.04, student_temp=0.1)

    def compute_training_loss(self, args, student_network, batch, train_info):
        if train_info.current_step == 0:
            initialize_teacher_network(args, student_network, self.teacher_network)
            self.center = torch.rand(4096).to(args.device)
            if is_distributed():
                torch.distributed.broadcast(self.center, src=0)
            self.center.requires_grad = False
        else:
            momentum = get_momentum(self.base_momentum, train_info.current_step, train_info.total_steps)
            update_teacher_network(student_network, self.teacher_network, momentum)

        views_1 = batch[0][0].to(args.device)
        views_2 = batch[0][1].to(args.device)

        s1, s2 = torch.chunk(student_network(torch.cat([views_1, views_2])), 2)

        with torch.no_grad():
            t1, t2 = torch.chunk(self.teacher_network(torch.cat([views_1, views_2])), 2)

        if train_info.current_epoch <= 30:
            # warmup teacher's temperature from 0.04 to 0.07 in the first 30 epochs
            self.distill_criterion.update_teacher_temp(0.04 + 0.001 * train_info.current_epoch)
        loss = 0.5 * (self.distill_criterion(t1, s2, self.center) + self.distill_criterion(t2, s1, self.center))

        # update center vector
        with torch.no_grad():
            new_center = torch.cat([t1, t2]).sum(dim=0)
            n = (torch.tensor(t1.shape[0]) + torch.tensor(t2.shape[0])).to(args.device)
            if is_distributed():
                torch.distributed.all_reduce(new_center)
                torch.distributed.all_reduce(n)
            new_center.div_(n)
            self.center = 0.9 * self.center + 0.1 * new_center

        return loss

    def init_metrics(self):
        raise NotImplementedError

    def update_metrics(self, args, model, batch):
        raise NotImplementedError

    def summarize_metrics(self):
        raise NotImplementedError


def main():
    parser = get_argument_parser(distributed=True)

    # Add custom arguments
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Directory consisting of training images."
    )
    parser.add_argument(
        "--base_momentum",
        type=float,
        default=0.996,
        help="Base momentum parameter for the update of teacher network."
    )
    parser.add_argument(
        "--save_teacher",
        action='store_true',
        help="Whether to save the final teacher model."
    )

    args = parse_and_process_arguments(parser)

    student_network = DINOModel(n_channels=3)
    teacher_network = DINOModel(n_channels=3)
    train_dataset = PreTrainMultiviewDataset(args.root_dir)

    callback = PretrainCallback(teacher_network, args.base_momentum)
    job_runner = JobRunner(args, student_network, callback)
    job_runner.run(train_dataset, None, pin_memory=False, train_transforms_image=TRANSFORM_TRAIN_CONTRASTIVE)
    if args.save_teacher:
        torch.save(teacher_network.state_dict(), os.path.join(args.model_dir, "teacher_network.pt"))


if __name__ == "__main__":
    main()
