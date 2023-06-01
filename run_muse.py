#!/usr/bin/env python3
import datetime

from pathlib import Path
from collections import defaultdict
import fire
from typing import Dict, List, Any, Union, Sequence, Tuple
import logging

import torch
import numpy as np
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Engine, Events)
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    CosineAnnealingScheduler,
    global_step_from_engine,
)
import ignite
import ignite.distributed as idist
from ignite.metrics import RunningAverage, Loss, EpochMetric
from ignite.utils import convert_tensor
from torch.cuda.amp import GradScaler, autocast
from ignite.utils import manual_seed, setup_logger

import dataset
import models
import utils
import criterion as losses

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")

def transfer_to_device(batch):
    DEVICE = idist.device()
    return (x.to(DEVICE, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)


def __setup(config: Path,
            default_args=utils.DEFAULT_ARGS,
            **override_kwargs) -> Dict[str, Any]:
    config_parameters = utils.parse_config_or_kwargs(
        config, default_args=default_args, **override_kwargs)
    
    return config_parameters


def log_basic_info(logger, config_parameters):
    logger.info(f"Train {config_parameters['model']} on VoxCeleb2-2Mix")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
    # explicitly import cudnn as
    # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")
    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config_parameters.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")

def train(local_rank, config_parameters):
    """Trains a given model specified in the config file or passed as the --model parameter.
    All options in the config file can be overwritten as needed by passing --PARAM
    Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'

    :param config: yaml config file
    :param **kwargs: parameters to overwrite yaml config
    """
    rank = idist.get_rank()
    manual_seed(config_parameters["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name='vox2_ddp')
    

    outputpath = config_parameters['outputpath']
    if rank == 0:
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{config_parameters['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        output_dir = Path(outputpath) / folder_name
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        config_parameters["outputpath"] = output_dir.as_posix()
        log_file = output_dir / 'train.log'
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
        logger.info(f"Output dir: {config_parameters['outputpath']}")
        log_basic_info(logger, config_parameters)

        if "cuda" in device.type:
            config_parameters["cuda device name"] = torch.cuda.get_device_name(local_rank)
  
    train_df = config_parameters['train_scp']
    cv_df = config_parameters['cv_scp']
    train_dur = config_parameters['train_dur']
    cv_dur = config_parameters['cv_dur']
    
    train_ds = dataset.Vox2_Dataset(train_df, train_dur, batch_size = config_parameters['batch_size'])
    cv_ds = dataset.Vox2_Dataset(cv_df, cv_dur, batch_size = config_parameters['batch_size'])
    nproc = idist.get_nproc_per_node() 
    train_loader = idist.auto_dataloader(train_ds, batch_size=nproc, num_workers=config_parameters['num_workers'] * nproc, shuffle=True, drop_last=True, collate_fn = dataset.dummy_collate_fn)
    cv_loader = idist.auto_dataloader(cv_ds, batch_size=nproc, num_workers=config_parameters['num_workers'] * nproc, shuffle=False, drop_last=True, collate_fn = dataset.dummy_collate_fn)
    config_parameters['num_iters_per_epoch'] = len(train_loader)

    model = getattr(models, config_parameters['model'])()
    model = idist.auto_model(model, sync_bn = True)

    optimizer = getattr(
        torch.optim,
        config_parameters['optimizer'],
    )(model.parameters(), **config_parameters['optimizer_args'])
    optimizer = idist.auto_optim(optimizer)


    loss_func = getattr(losses, config_parameters['loss'])().to(idist.device())

    epochs = config_parameters['epochs']
    decay_steps = epochs * config_parameters['num_iters_per_epoch']

    trainer = create_trainer(model, optimizer, loss_func, train_loader.sampler, config_parameters, logger)
    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = {
            "si-snr loss": Loss(loss_func.si_snr_loss)
            }

    evaluator = create_evaluator(model, metrics=metrics, config_parameters=config_parameters)

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = evaluator.run(cv_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Val", state.metrics)
    
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda _: logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}"))
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=config_parameters["validate_every"]) | Events.COMPLETED, run_validation)

    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(config_parameters),
        filename_prefix="best",
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
        score_name="val_si_snr",
        score_function=Checkpoint.get_default_score_fn("si-snr loss", -1.0),
    )

    # lr scheduler
    half_lr_rate_handler = utils.half_lr_rate(optimizer = optimizer, param_name = 'lr', score_function = Checkpoint.get_default_score_fn("si-snr loss", -1.0), patience = 3)
    evaluator.add_event_handler(Events.COMPLETED, half_lr_rate_handler) 

    evaluator.add_event_handler(Events.COMPLETED(lambda *_: trainer.state.epoch > 30), best_model_handler)

    earlystop_handler = EarlyStopping(
            patience=config_parameters.get('early_stop', 10),
            score_function=Checkpoint.get_default_score_fn('si-snr loss', -1.0),
            trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)
        
    try:
        trainer.run(train_loader, max_epochs=config_parameters["epochs"], epoch_length=config_parameters["epoch_length"])
    except Exception as e:
        logger.exception("")
        raise e


def create_trainer(model, optimizer, criterion, train_sampler, config_parameters, logger):
    with_amp = config_parameters["with_amp"]
    scaler = GradScaler(enabled=with_amp)
    
    def train_step(engine, batch):
        model.train()
        with autocast(enabled=with_amp):
            mixture, source ,condition, spkid = batch
            mixture, source, condition = transfer_to_device([mixture, source, condition])
            speaker = spkid.flatten().to(idist.device())
            pred_speaker, pred_wav = model(mixture, condition)
            loss = criterion(source, pred_wav, speaker, pred_speaker)
            total_loss = loss['si_snr'] + 0.1 * loss['ce']
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(optimizer)
        scaler.update()

        return {
            "si-snr loss": loss['si_snr'].item(),
            "speaker loss": loss['ce'].item()
        }
    trainer = Engine(train_step)
    trainer.logger = logger

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer}
    metric_names = [
        "si-snr loss",
        "speaker loss"
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        save_handler=get_save_handler(config_parameters),
        output_names=metric_names,
        with_pbars=True,
        clear_cuda_cache=False,
        log_every_iters=1
    )

    resume_from = config_parameters.get("resume_from", None)
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def create_evaluator(model, metrics, config_parameters, tag='val'):
    with_amp = config_parameters['with_amp']

    @torch.no_grad()
    def evaluate_step(engine, batch):
        model.eval()
        mixture, source ,condition, _ = batch
        mixture, source, condition = transfer_to_device([mixture, source, condition])
        with autocast(enabled=with_amp):
            _, pred_wav = model(mixture, condition)
        return source, pred_wav
    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    if idist.get_rank() == 0:
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)
            
    return evaluator

def run(config, **kwargs):
    setup_args = __setup(config, **kwargs)
    config_parameters = setup_args
    spawn_kwargs={"nproc_per_node": config_parameters['nproc_per_node']}
    with idist.Parallel(backend = config_parameters['backend'], **spawn_kwargs) as parallel:
        parallel.run(train, config_parameters)
   
def get_save_handler(config):
    return DiskSaver(config["outputpath"], require_empty=False)

if __name__ == "__main__":
    fire.Fire({"run": run})
