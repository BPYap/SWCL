import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from retinal_fundus_encoder.model import AVAILABLE_BACKBONES, AVAILABLE_BACKBONES_SEGMENTATION


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_main_process():
    return torch.distributed.get_rank() == 0 if is_distributed() else True


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name):
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self):
            super().__init__()

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg, file=sys.stderr)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[TqdmLoggingHandler()]
    )
    _logger = logging.getLogger(name)
    # _logger.addHandler(logging.StreamHandler(sys.stdout))  # send a copy of logging messages to stdout

    return _logger


logger = get_logger(__name__)


def get_argument_parser(segmentation=False, finetune=False, distributed=False):
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the model and checkpoints will be stored."
    )

    # Computation resources
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="Number of GPU(s) to use if available."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker(s) for data loader."
    )

    # Model architecture
    parser.add_argument(
        "--encoder_backbone",
        type=str,
        default='resnetv2-50x1',
        help=f"Name of the encoder backbone. Available options: "
        f"{AVAILABLE_BACKBONES if not segmentation else AVAILABLE_BACKBONES_SEGMENTATION}."
    )
    parser.add_argument(
        "--imagenet_init",
        action='store_true',
        help="If applicable, choose whether to initialize the encoder with weights pre-trained on ImageNet."
    )

    # Hyperparameters
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=200,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_epochs",
        default=0,
        type=int,
        help="Number of epochs to linearly warmup to."
    )
    parser.add_argument(
        "--lr_scheduler",
        default="cosine",
        type=str,
        help="Learning rate scheduler after linear warmup. Available options: ['none', 'cosine', 'step']"
    )
    parser.add_argument(
        "--step_decay_milestones",
        type=str,
        help="List of epoch milestones to apply learning rate decay, only applicable if 'step' is specified "
             "in `lr_scheduler`."
    )
    parser.add_argument(
        "--step_decay_factor",
        default=0.1,
        type=float,
        help="Multiplicative factor to be applied to the learning rate at each decay milestone, only applicable "
             "if 'step' is specified in `lr_scheduler`."
    )
    parser.add_argument(
        "--learning_rate",
        default=0.01,
        type=float,
        help="The initial learning rate."
    )
    parser.add_argument(
        "--optimizer",
        default="sgd",
        type=str,
        help="Optimization algorithm to use. Available options: ['sgd', 'adam']"
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum for SGD."
    )
    parser.add_argument(
        "--nesterov",
        action='store_true',
        help="Whether to use Nesterov SGD."
    )
    parser.add_argument(
        "--weight_decay",
        default=5e-4,
        type=float,
        help="Weight decay for regularization."
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.0,
        type=float,
        help="Dropout rate applied after each downscaling block."
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html"
    )

    # Logging
    parser.add_argument(
        "--logging_steps",
        default=-1,
        type=int,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        default=-1,
        type=int,
        help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action='store_true',
        help="Run evaluation during training at each logging step."
    )

    # Other arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory consisting of model checkpoint and/or optimizer and scheduler for resuming training."
    )
    parser.add_argument(
        "--pretrain_model_path",
        type=str,
        help="Path to pre-trained model for weights initialization."
    )
    parser.add_argument(
        "--overwrite_model_dir",
        action='store_true',
        help="Whether to overwrite the content of the model directory during training."
    )
    parser.add_argument(
        "--do_train",
        action='store_true',
        help="Whether to run training on train set."
    )
    parser.add_argument(
        "--do_eval",
        action='store_true',
        help="Whether to run evaluation on dev set."
    )
    parser.add_argument(
        "--random_validation",
        action='store_true',
        help="Create a validation set by randomly splitting 20% of the training samples. This option has no effect"
             "if an evaluation set is already provided."
    )

    if segmentation:
        parser.add_argument(
            "--segmentation_architecture",
            type=str,
            default='u-net',
            help="Name of the segmentation architecture. Available options: ['u-net', 'deeplabv3+']."
        )

    if finetune:
        parser.add_argument(
            "--freeze_weights",
            action="store_true",
            help="Whether to freeze pre-trained weights during fine-tuning, "
                 "only applicable if `--pretrain_model_path` is specified."
        )

    if distributed:
        parser.add_argument(
            "--world_size",
            type=int,
            help="Number of distributed processes."
        )
        parser.add_argument(
            "--shared_file_path",
            type=str,
            help="Absolute path to store the shared file during distributed training."
        )
        parser.add_argument(
            "--rank",
            type=int,
            help="Rank of the current process."
        )
        parser.add_argument(
            "--local_rank",
            type=int,
            help="Local Device ID."
        )

    return parser


def parse_and_process_arguments(parser):
    args = parser.parse_args()

    if (
            os.path.exists(args.model_dir)
            and os.listdir(args.model_dir)
            and args.do_train
            and not args.overwrite_model_dir
    ):
        raise ValueError(
            "Model directory ({}) already exists and is not empty. Use --overwrite_model_dir to overcome.".format(
                args.model_dir
            )
        )
    elif is_main_process() and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Setup CPU/GPU
    device = torch.device("cpu")
    if hasattr(args, "world_size") and args.world_size:  # distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"file://{args.shared_file_path}",
            rank=args.rank,
            world_size=args.world_size
        )
        args.n_gpu = 1

        if not is_main_process():
            logger.setLevel(logging.WARN)
    elif args.n_gpu >= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(args.n_gpu)])
        gpu_count = torch.cuda.device_count()
        if gpu_count < args.n_gpu:
            logger.warning(f"Only {gpu_count} GPU(s) are available but {args.n_gpu} is requested.")
            args.n_gpu = gpu_count
        if torch.cuda.is_available():
            device = torch.device("cuda")
    args.device = device

    _set_seed(args.seed)

    logger.warning("Device: %s, n_gpu: %s, 16-bits training: %s", device, args.n_gpu, args.fp16)
    logger.info("***** Training/evaluation parameters *****")
    for parameter, value in vars(args).items():
        logger.info(f"  {parameter} = {value}")

    return args
