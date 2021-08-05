import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from retinal_fundus_encoder.config import get_argument_parser, parse_and_process_arguments
from retinal_fundus_encoder.dataset import PreTrainMultiviewDataset
from retinal_fundus_encoder.model import BYOLModel
from retinal_fundus_encoder.train_utils import Callback, JobRunner
from retinal_fundus_encoder.transforms import TRANSFORM_TRAIN_CONTRASTIVE


def initialize_target_network(args, online_network, target_network):
    target_network.init(args.encoder_backbone, imagenet_init=False)
    target_network.to(args.device)
    update_target_network(online_network, target_network, momentum=0)


def update_target_network(online_network, target_network, momentum):
    if hasattr(online_network, "module"):
        online_network = online_network.module

    # encoder parameters
    for online_param, target_param in zip(online_network.encoder.parameters(),
                                          target_network.encoder.parameters()):
        target_param.data = momentum * target_param.data + (1 - momentum) * online_param.data
        target_param.requires_grad = False

    # projector parameters
    for online_param, target_param in zip(online_network.projection_network.parameters(),
                                          target_network.projection_network.parameters()):
        target_param.data = momentum * target_param.data + (1 - momentum) * online_param.data
        target_param.requires_grad = False


def get_momentum(base_momentum, current_step, total_steps):
    return 1 - (1 - base_momentum) * (math.cos((current_step * math.pi) / total_steps) + 1) / 2


class PretrainCallback(Callback):
    def __init__(self, target_network, base_momentum):
        self.target_network = target_network
        self.base_momentum = base_momentum

        self.regression_criterion = None

    def init_training_criteria(self):
        self.regression_criterion = nn.MSELoss(reduction='sum')

    def compute_training_loss(self, args, model, batch, train_info):
        if train_info.current_step == 0:
            initialize_target_network(args, model, self.target_network)
        else:
            momentum = get_momentum(self.base_momentum, train_info.current_step, train_info.total_steps)
            update_target_network(model, self.target_network, momentum)

        views_1 = batch[0][0].to(args.device)
        views_2 = batch[0][1].to(args.device)
        batch_size = len(views_1)

        views_1_predictions, views_2_predictions = torch.chunk(model(torch.cat([views_1, views_2])), 2)

        with torch.no_grad():
            views_1_targets, views_2_targets = torch.chunk(self.target_network(torch.cat([views_1, views_2])), 2)

        loss = (self.regression_criterion(F.normalize(views_1_predictions), F.normalize(views_2_targets)) +
                self.regression_criterion(F.normalize(views_2_predictions), F.normalize(views_1_targets))) / batch_size

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
        help="Base momentum parameter for the update of target network."
    )

    args = parse_and_process_arguments(parser)

    online_network = BYOLModel(n_channels=3, attach_predictor=True)
    target_network = BYOLModel(n_channels=3, attach_predictor=False)
    train_dataset = PreTrainMultiviewDataset(args.root_dir)

    callback = PretrainCallback(target_network, args.base_momentum)
    job_runner = JobRunner(args, online_network, callback)
    job_runner.run(train_dataset, None, pin_memory=False, train_transforms_image=TRANSFORM_TRAIN_CONTRASTIVE)


if __name__ == "__main__":
    main()
