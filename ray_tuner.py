import os
from typing import Any, Dict, Type

import dgl
import torch
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from ray import tune
from ray.tune import ExperimentAnalysis

from datamodule import ContextualDataModule
from contextual_model import ContextualModel
from callbacks import (
    get_loggers_and_callbacks,
    reporter,
    scheduler,
    search_alg,
)


def update_all_ray_parameters_block_layer(parameters: Dict[str, Any], ray_parameters: Dict[str, Any]) -> Dict[str, Any]:
    for block_layer, contexts_params in parameters.items():
        if f"ray_{block_layer}" not in ray_parameters:
            continue
        ray_context = list(ray_parameters[f"ray_{block_layer}"].keys())[0]
        in_feats = contexts_params[ray_context[len("ray_") :]]["in_feats"]
        for context in contexts_params:
            parameters[block_layer][context]["in_feats"] = in_feats
    return parameters


def train_tune(
    config: Dict[str, Any],
    LightModel: Type[LightningModule],
    DataModule: Type[LightningDataModule],
    graph: dgl.DGLHeteroGraph,
    **kwargs: Any,
) -> None:
    # Define Environment's Variables
    os.environ["DGLBACKEND"] = "pytorch"

    # Empty GPUs' cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Ensure Reproducibility
    seed_everything(
        seed=123,
        workers=True,
    )

    if "ray_parameters_per_block_layer" in config:
        config["parameters_per_block_layer"] = update_all_ray_parameters_block_layer(
            parameters=config["parameters_per_block_layer"],
            ray_parameters=config["ray_parameters_per_block_layer"],
        )

    # Initialize DataModule and Model
    pl_datamodule = DataModule(graph, config)
    pl_model = LightModel(config, **kwargs)

    loggers, callbacks = get_loggers_and_callbacks(config=config, with_tune_report_callback=True)

    # Define Trainer
    trainer = Trainer(
        gpus=config.get("gpus"),
        strategy=config.get("strategy"),
        max_epochs=config.get("max_epochs"),
        logger=list(loggers.values()),
        callbacks=list(callbacks.values()),
        # profiler="simple",
        benchmark=False,
        num_sanity_val_steps=-1,
        enable_progress_bar=True,
        deterministic=True,
        # log_every_n_steps=5,
    )

    # Fit
    trainer.fit(
        model=pl_model,
        datamodule=pl_datamodule,
    )

    # Validate
    best_loss_path = None
    loss_checkpoint_callback = callbacks["loss_checkpoint"]
    if isinstance(loss_checkpoint_callback, ModelCheckpoint):
        if loss_checkpoint_callback.monitor == "total_loss_valid":
            best_loss_path = loss_checkpoint_callback.best_model_path
            print(f"Best model path based on loss: {best_loss_path}")
    trainer.validate(
        datamodule=pl_datamodule,
        model=pl_model,
        ckpt_path=best_loss_path,
    )


def tune_runner(
    config: Dict[str, Any],
    graph: dgl.DGLHeteroGraph,
    **kwargs: Any,
) -> ExperimentAnalysis:
    if config.get("model_name") == "uk-multi-context":
        Model: Type[LightningModule] = ContextualModel
    else:
        raise NotImplementedError(f"There is no implementation for the model name: {config.get('model_name')}")
    # Model: Type[LightningModule] = ContextualModel
    experiment_analysis: ExperimentAnalysis = tune.run(
        run_or_experiment=tune.with_parameters(
            trainable=train_tune,
            LightModel=Model,
            DataModule=ContextualDataModule,
            graph=graph,
            **kwargs,
        ),
        config=config,
        search_alg=search_alg(config),
        scheduler=scheduler(config),
        resume=config.get("resume"),
        name=config.get("name_log"),
        metric=config.get("hyper_search_variable_to_check"),
        mode=config.get("hyper_search_variable_to_check_mode"),
        resources_per_trial={
            "cpu": config.get("cpu_per_trial"),
            "gpu": config.get("gpu_per_trial"),
        },
        local_dir=config.get("experiment_log_dir"),
        num_samples=config.get("n_trials"),
        progress_reporter=reporter(config),
        max_failures=config.get("number_of_retry_per_fail"),
        reuse_actors=config.get("reuse_actors"),
        max_concurrent_trials=config.get("max_concurrent_trials"),
    )
    return experiment_analysis
