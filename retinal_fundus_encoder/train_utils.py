# Codes inspired and adapted from https://github.com/huggingface/transformers

import json
import math
import os
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from retinal_fundus_encoder.config import logger, is_distributed, is_main_process
from retinal_fundus_encoder.transforms import TransformsDataset, DEFAULT_TRANSFORM


def _load_model_state(model, model_path, strict=True):
    logger.info(f"Loading model parameters from '{model_path}'")
    file_name = os.path.basename(model_path)
    if file_name in ["BiT-S-R50x1.npz", "BiT-M-R50x1.npz"]:
        logger.info(f"Loading from BiT checkpoint")
        loaded_states = np.load(model_path)
        model.encoder.load_from(loaded_states)
    else:
        # generic state loading
        loaded_states = torch.load(model_path, map_location='cpu')
        logger.info(model.load_state_dict(loaded_states, strict=strict))


def _freeze_module(module):
    names = []
    for name, param in module.named_parameters():
        param.requires_grad = False
        names.append(name)
    logger.info(f"Number of frozen layers = {len(names)}")
    logger.info(f"Frozen layers: {names}")


def get_parameter_names(model, forbidden_layer_types):
    """ Returns the names of the model parameters that are not inside a forbidden layer.
    Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_pt_utils.py#L964
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())

    return result


class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    """Linearly warmup learning rate to a specified update steps then apply cosine annealing.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of update steps in linear warmup.
        total_steps (int): Total number of update steps.
    """

    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        super().__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.warmup_steps > 0 and self.last_epoch <= self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            offset_steps = self.last_epoch - self.warmup_steps
            offset_total = self.total_steps - self.warmup_steps
            return [base_lr * (1 + math.cos(math.pi * offset_steps / offset_total)) / 2
                    for base_lr in self.base_lrs]


class WarmupStepLR(optim.lr_scheduler._LRScheduler):
    """Linearly warmup learning rate to a specified update steps then apply learning rate decay at specified steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of update steps in linear warmup.
        decay_steps (list): List of steps to apply learning rate decay.
        factor (float): Factor of learning rate decay.
    """

    def __init__(self, optimizer, warmup_steps, decay_steps, factor):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.factor = factor
        self._factor = 1

        super().__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.warmup_steps > 0 and self.last_epoch <= self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            if self.last_epoch in self.decay_steps:
                self._factor *= self.factor

            return [base_lr * self._factor for base_lr in self.base_lrs]


class TrainInfo:
    """ Container to store training info at each update step.
    """

    def __init__(self, current_epoch, current_step, total_steps):
        self.current_epoch = current_epoch
        self.current_step = current_step
        self.total_steps = total_steps


class Callback:
    def init_training_criteria(self):
        """ Initialize criteria/objectives for loss computation during training.
        """
        raise NotImplementedError("`init_training_criteria` is not implemented.")

    def compute_training_loss(self, args, model, batch, train_info):
        """ Compute training loss given a training batch.
        Args:
            args (Namespace): Parsed command-line arguments.
            model (UNet): Model to train.
            batch (tuple): Tuple of input and target tensors.
            train_info (TrainInfo): Training info at current update step."
        """
        raise NotImplementedError("`compute_training_loss` is not implemented.")

    def init_metrics(self):
        """ Initialize metrics for evaluation.
        """
        raise NotImplementedError("`init_metrics` is not implemented.")

    def update_metrics(self, args, model, batch):
        """ Update metrics given an evaluation batch.
        Args:
            args (Namespace): Parsed command-line arguments.
            model (UNet): Model to evaluate.
            batch (tuple): Tuple of input and target tensors.
        """
        raise NotImplementedError("`update_metrics` is not implemented.")

    def summarize_metrics(self):
        """ Extract metrics of interest into a dictionary with metric names as keys and metrics as values.
        """
        raise NotImplementedError("`summarize_metrics` is not implemented.")


class JobRunner:
    def __init__(self, args, model, callback):
        self.args = args
        self.model = model
        self.callback = callback

        # initialize model
        if hasattr(args, 'segmentation_architecture'):
            self.model.init(args.segmentation_architecture, args.encoder_backbone, args.imagenet_init)
        else:
            self.model.init(args.encoder_backbone, args.imagenet_init)
        if hasattr(self.model, 'set_dropout_rate'):
            self.model.set_dropout_rate(args.dropout_rate)

        # load from existing model states
        if args.do_eval and not args.do_train and os.path.exists(args.model_dir):
            # Load states from trained model for evaluation
            model_path = os.path.join(args.model_dir, 'pytorch_model.bin')
            _load_model_state(self.model, model_path, strict=True)
        elif args.do_train and args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
            # Load states from checkpoint for resuming training
            checkpoint_path = os.path.join(args.checkpoint_dir, 'pytorch_model.bin')
            _load_model_state(self.model, checkpoint_path, strict=True)
        elif args.pretrain_model_path and os.path.exists(args.pretrain_model_path):
            # Load states from pretrained encoder for fine-tuning
            _load_model_state(self.model, args.pretrain_model_path, strict=False)
            if hasattr(args, 'freeze_weights') and args.freeze_weights:
                # freeze encoder weights
                _freeze_module(self.model.encoder)

        self.model.to(args.device)

    def train_model(self, train_data_loader, eval_data_loader=None):
        args = self.args
        model = self.model
        model.train()

        self.callback.init_training_criteria()

        # Calculate number of training steps
        steps_per_epoch = len(train_data_loader) // args.gradient_accumulation_steps
        if args.max_steps > 0:
            total_steps = args.max_steps
            args.num_train_epochs = args.max_steps // steps_per_epoch + 1
        else:
            total_steps = steps_per_epoch * args.num_train_epochs

        # Disable weight decay for normalization parameters and biases
        decay_parameters = get_parameter_names(
            model,
            forbidden_layer_types=[torch.nn.GroupNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d]
        )
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        # Initialize optimizer
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                momentum=args.momentum,
                nesterov=args.nesterov
            )
        elif args.optimizer == 'adam':
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: '{args.optimizer}'.")

        # Initialize learning rate scheduler
        if args.lr_scheduler == 'cosine':
            lr_scheduler = WarmupCosineLR(optimizer, args.warmup_epochs * steps_per_epoch, total_steps)
        elif args.lr_scheduler == 'step':
            epoch_milestones = [int(e) for e in args.step_decay_milestones.split(',')]
            step_milestones = [e * steps_per_epoch for e in epoch_milestones]
            factor = args.step_decay_factor
            lr_scheduler = WarmupStepLR(optimizer, args.warmup_epochs * steps_per_epoch, step_milestones, factor)
        elif args.lr_scheduler == 'none':
            lr_scheduler = WarmupStepLR(optimizer, args.warmup_epochs * steps_per_epoch, [], 1)
        else:
            raise ValueError(f"Unknown learning rate scheduler: '{args.lr_scheduler}'.")

        # Load optimizer and scheduler states if exist
        if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
            optimizer_path = os.path.join(args.checkpoint_dir, "optimizer.pt")
            scheduler_path = os.path.join(args.checkpoint_dir, "scheduler.pt")
            if os.path.isfile(optimizer_path) and os.path.isfile(scheduler_path):
                logger.info(f"Loading existing optimizer and scheduler states from checkpoint "
                            f"directory '{args.checkpoint_dir}'")
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=args.device))
                lr_scheduler.load_state_dict(torch.load(scheduler_path))

        # Use mixed precision training if specified
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (must be put after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        elif is_distributed():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )

        # Train
        logger.info("***** Running training *****")
        logger.info("  Num samples = %d", len(train_data_loader.dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Batch size per device = %d", args.per_device_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info(
            "  Total train batch size (with parallel, distributed & accumulation) = %d",
            args.per_device_train_batch_size
            * max(1, args.n_gpu)
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if is_distributed() else 1)
        )
        logger.info("  Total optimization steps = %d", total_steps)

        global_step = 0

        # Check if continuing training from a checkpoint
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        if args.checkpoint_dir and os.path.exists(args.checkpoint_dir) and args.checkpoint_dir.split("-")[-1].isdigit():
            # set global_step to global_step of last saved checkpoint
            global_step = int(args.checkpoint_dir.split("-")[-1])
            epochs_trained = global_step // (len(train_data_loader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_data_loader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        training_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, args.num_train_epochs, desc="Epoch", disable=not is_main_process()
        )
        for epoch in train_iterator:
            if is_distributed():
                train_data_loader.sampler.set_epoch(epoch)
            epoch_iterator = tqdm(
                train_data_loader, desc="Iteration", leave=False, disable=not is_main_process()
            )
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # Run backpropagation
                train_info = TrainInfo(epoch, global_step, total_steps)
                loss = self.callback.compute_training_loss(args, model, batch, train_info)
                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Update weights
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    training_loss += loss.item()

                    # Logging
                    if is_main_process() and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logger.info("=" * 80)
                        last_lr = lr_scheduler.get_last_lr()[0]
                        train_summary = {
                            "step": global_step,
                            "learning rate": f"{last_lr:.3E}",
                            "training loss": f"{loss.item():.3E}"
                        }
                        logger.info("  [Training log]")
                        for stat, value in train_summary.items():
                            logger.info(f"    {stat:<15} = {value}")

                        if args.evaluate_during_training:
                            eval_summary = self.evaluate_model(eval_data_loader)
                            model.train()  # revert back to training mode
                            logger.info("  [Evaluation log]")
                            for name, stat in eval_summary.items():
                                logger.info(f"    {name:<15} = {stat}")

                    # Save model checkpoint
                    if is_main_process() and args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.model_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        logger.info("Saving model checkpoint to '%s'", output_dir)
                        model_to_save = model.module if hasattr(model, "module") else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                        logger.info("Saving optimizer and scheduler states to '%s'", output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if 0 < args.max_steps <= global_step:
                    epoch_iterator.close()
                    break

            if 0 < args.max_steps <= global_step:
                train_iterator.close()
                break

        return global_step, training_loss / global_step

    def evaluate_model(self, eval_data_loader):
        args = self.args
        model = self.model
        model.eval()

        self.callback.init_metrics()

        if not args.do_train:
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            elif is_distributed():
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=True
                )

        # Eval
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_data_loader.dataset))
        logger.info("  Batch size per device = %d", args.per_device_eval_batch_size)

        with torch.no_grad():
            for batch in tqdm(eval_data_loader, desc="Evaluation", leave=False):
                self.callback.update_metrics(args, model, batch)

        eval_summary = self.callback.summarize_metrics()

        return eval_summary

    def run(self, train_dataset, eval_dataset, pin_memory=False,
            train_transforms_image=DEFAULT_TRANSFORM, train_transforms_mask=None,
            eval_transforms_image=DEFAULT_TRANSFORM, eval_transforms_mask=None):
        args = self.args

        if not args.do_train and not args.do_eval:
            logger.info("Neither `do_train` nor `do_eval` is specified. Terminating...")
            exit(1)

        if train_dataset and not eval_dataset and args.random_validation:
            eval_size = int(0.2 * len(train_dataset))
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset=train_dataset,
                lengths=[len(train_dataset) - eval_size, eval_size]
            )

        train_data_loader = None
        if train_dataset:
            train_dataset = TransformsDataset(train_dataset, train_transforms_image, train_transforms_mask)
            train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
            if is_distributed():
                train_sampler = DistributedSampler(train_dataset)
            else:
                train_sampler = torch.utils.data.RandomSampler(train_dataset)

            train_data_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=pin_memory
            )

        eval_data_loader = None
        if eval_dataset:
            eval_dataset = TransformsDataset(eval_dataset, eval_transforms_image, eval_transforms_mask)
            eval_data_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=args.per_device_eval_batch_size * max(1, args.n_gpu),
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory
            )

        logs = dict()
        logger.info("=" * 80)
        if self.args.do_train and train_data_loader:
            global_step, tr_loss = self.train_model(train_data_loader, eval_data_loader)

            logs.update({"global_step": global_step, "training_loss": tr_loss})
            logger.info("=" * 80)
            logger.info("  [Training summary]")
            logger.info("    global_step  = %s", global_step)
            logger.info("    average loss = %s", tr_loss)

            # Save model
            if is_main_process():
                # save training arguments
                with open(os.path.join(args.model_dir, "training_args.json"), 'w') as f:
                    args_dict = vars(args)
                    args_dict['device'] = str(args_dict['device'])
                    json.dump(args_dict, f, indent=4)
                # save model states
                logger.info("Saving model to '%s'", args.model_dir)
                torch.save(self.model.state_dict(), os.path.join(args.model_dir, "pytorch_model.bin"))

        if is_main_process() and args.do_eval and eval_data_loader:
            eval_summary = self.evaluate_model(eval_data_loader)

            if eval_summary:
                logs.update(eval_summary)
                logger.info("  [Evaluation summary]")
                for name, stat in eval_summary.items():
                    logger.info(f"    {name:<15} = {stat}")

        return logs
