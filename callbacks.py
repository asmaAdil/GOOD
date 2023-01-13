import argparse
import logging
import os
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from ray import tune
from ray.tune import CLIReporter, ProgressReporter
from ray.tune.integration.pytorch_lightning import (
    TuneCallback,
    TuneReportCallback,
    _TuneCheckpointCallback,
)
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from ray.tune.suggest import Searcher
from ray.tune.suggest.hyperopt import HyperOptSearch

logger = logging.getLogger(__name__)


class TuneReportCallbackWithSanityChecking(TuneReportCallback):
    def _get_report_dict(self, trainer: Trainer, pl_module: LightningModule) -> Optional[Dict[str, Any]]:
        # Don't report if just doing initial validation sanity checks.
        if trainer.sanity_checking:
            return None
        if not self._metrics:
            report_dict = {k: v.item() for k, v in trainer.callback_metrics.items()}
        else:
            report_dict = {}
            for key in self._metrics:
                if isinstance(self._metrics, dict):
                    metric = self._metrics[key]
                else:
                    metric = key
                if metric in trainer.callback_metrics:
                    report_dict[key] = trainer.callback_metrics[metric].item()
                else:
                    logger.warning("Metric %s does not exist in " "`trainer.callback_metrics.", metric)

        return report_dict


class _TuneCheckpointCallbackWithSanityChecking(_TuneCheckpointCallback):
    def _handle(self, trainer: Trainer, pl_module: Optional[LightningModule]) -> None:
        if trainer.sanity_checking:
            return
        step = f"epoch={trainer.current_epoch}-step={trainer.global_step}"
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            trainer.save_checkpoint(os.path.join(checkpoint_dir, self._filename))


class TuneReportCheckpointCallback(TuneCallback):
    """PyTorch Lightning report and checkpoint callback

    Saves checkpoints after each validation step. Also reports metrics to Tune,
    which is needed for checkpoint registration.

    Args:
        metrics (str|list|dict): Metrics to report to Tune. If this is a list,
            each item describes the metric key reported to PyTorch Lightning,
            and it will reported under the same name to Tune. If this is a
            dict, each key will be the name reported to Tune and the respective
            value will be the metric key reported to PyTorch Lightning.
        filename (str): Filename of the checkpoint within the checkpoint
            directory. Defaults to "checkpoint".
        on (str|list): When to trigger checkpoint creations. Must be one of
            the PyTorch Lightning event hooks (less the ``on_``), e.g.
            "batch_start", or "train_end". Defaults to "validation_end".


    Example:

    .. code-block:: python

        import pytorch_lightning as pl
        from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

        # Save checkpoint after each training batch and after each
        # validation epoch.
        trainer = pl.Trainer(
            callbacks=[
                TuneReportCheckpointCallback(
                    metrics={"loss": "val_loss", "mean_accuracy": "val_acc"},
                    filename="trainer.ckpt",
                    on="validation_end",
                )
            ]
        )


    """

    _checkpoint_callback_cls = _TuneCheckpointCallbackWithSanityChecking
    _report_callbacks_cls = TuneReportCallbackWithSanityChecking

    def __init__(
        self,
        metrics: Union[None, str, List[str], Dict[str, str]] = None,
        filename: str = "checkpoint",
        on: Union[str, List[str]] = "validation_end",
    ) -> None:
        super().__init__(on)
        self._checkpoint = self._checkpoint_callback_cls(filename, on)
        self._report = self._report_callbacks_cls(metrics, on)

    def _handle(self, trainer: Trainer, pl_module: Optional[LightningModule]) -> None:
        self._checkpoint._handle(trainer, pl_module)  # pylint: disable=protected-access
        # Purpose of this part is to debug the conflict that exist between Ray and PyTorch-Lightning
        # This method has been copied and pasted
        # so we can update it with the correct versions of Checkpoint and Reporting
        assert pl_module is not None
        self._report._handle(trainer, pl_module)  # pylint: disable=protected-access
        # Purpose of this part is to debug the conflict that exist between Ray and PyTorch-Lightning
        # This method has been copied and pasted
        # so we can update it with the correct versions of Checkpoint and Reporting


class TrackingLogger(LightningLoggerBase):
    def __init__(self, experiment: Optional[Dict[str, List[float]]] = None, version: Optional[int] = None):
        super().__init__()

        self._tb_version = version

        if experiment is not None:
            self._experiment: Dict[str, List[float]] = experiment
        self._experiment = {}

    @property  # type: ignore  # Copied from PyTorch-Lightning documentation
    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#make-a-custom-logger
    @rank_zero_experiment
    def experiment(self) -> Any:
        # Return the experiment object associated with this logger.
        return self._experiment

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        if step is not None:
            self._experiment.setdefault("step", []).append(step)
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self._experiment.setdefault(key, {}).setdefault(sub_key, []).append(sub_value)
            else:
                self._experiment.setdefault(key, []).append(value)

    @rank_zero_only
    def log_hyperparams(self, params: argparse.Namespace, *args: Any, **kwargs: Any) -> None:
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    def log_text(self, *args: Any, **kwargs: Any) -> None:
        """
        Need to implement this abstract method.
        It is useful in case you want to log some text.

        :param args:
        :param kwargs:
        :return:
        """

    def log_image(self, *args: Any, **kwargs: Any) -> None:
        """
        Need to implement this abstract method.
        It is useful in case you want to log an image.

        :param args:
        :param kwargs:
        :return:
        """

    @property
    def name(self) -> str:
        return "TrackingLogger"

    @property
    def version(self) -> Union[int, str]:
        # Return the experiment version, int or str.
        return self._tb_version if self._tb_version is not None else "0.x"

    @rank_zero_only
    def save(self) -> None:
        """
        Optional. Any code necessary to save logger data goes here
        If you implement this, remember to call `super().save()`
        at the start of the method (important for aggregation of metrics)

        :return:
        """
        super().save()

    @rank_zero_only
    def finalize(self, status: Any) -> None:
        """
        Optional. Any code that needs to be run after training
        finishes goes here

        :param status:
        :return:
        """


def get_loggers_and_callbacks(
    config: Dict[str, Any], with_tune_report_callback: bool = False
) -> Tuple[Dict[str, LightningLoggerBase], Dict[str, Callback]]:
    experiment_log_dir = Path(str(config.get("experiment_log_dir")))
    assert isinstance(experiment_log_dir, (str, Path))

    time_version = str(time()).replace(".", "_")
    (experiment_log_dir / time_version).mkdir(parents=True, exist_ok=True)

    tensorboard_logger = TensorBoardLogger(
        save_dir=str(experiment_log_dir / time_version),
        name="contextual_tensorboard_logs",
        prefix="",
        default_hp_metric=False,
        log_graph=False,
    )

    version = Path(f"version_{time_version}")
    print(f"Version: {version}")
    version /= "best_k_model_checkpoints"

    tracking_logger = TrackingLogger(
        experiment=None,
        version=int(version.parent.name.split("_")[-1]),
    )

    patience = config.get("patience")
    assert isinstance(patience, int)
    early_stopping = EarlyStopping(
        monitor="total_loss_valid",
        mode="min",
        patience=patience,
        strict=True,
    )
    loss_checkpoint_callback = ModelCheckpoint(
        monitor="total_loss_valid",
        mode="min",
        verbose=True,
        save_last=True,
        save_top_k=3,
        filename="contextual-{epoch:02d}-{total_loss_valid:.7f}",
        dirpath=Path(experiment_log_dir) / time_version / "contextual_tensorboard_logs" / version / "loss",
    )
    accuracy_checkpoint_callback = ModelCheckpoint(
        monitor="total_accuracy_valid",
        mode="max",
        verbose=True,
        save_last=True,
        save_top_k=3,
        filename="contextual-{epoch:02d}-{total_accuracy_valid:.4f}",
        dirpath=Path(experiment_log_dir) / time_version / "contextual_tensorboard_logs" / version / "accuracy",
    )
    auc_checkpoint_callback = ModelCheckpoint(
        monitor="total_auc_valid",
        mode="max",
        verbose=True,
        save_last=True,
        save_top_k=3,
        filename="contextual-{epoch:02d}-{total_auc_valid:.4f}",
        dirpath=Path(experiment_log_dir) / time_version / "contextual_tensorboard_logs" / version / "auc",
    )
    f1_checkpoint_callback = ModelCheckpoint(
        monitor="total_f1_valid",
        mode="max",
        verbose=True,
        save_last=True,
        save_top_k=3,
        filename="contextual-{epoch:02d}-{total_f1_valid:.4f}",
        dirpath=Path(experiment_log_dir) / time_version / "contextual_tensorboard_logs" / version / "f1",
    )
    ap_checkpoint_callback = ModelCheckpoint(
        monitor="total_ap_valid",
        mode="max",
        verbose=True,
        save_last=True,
        save_top_k=3,
        filename="contextual-{epoch:02d}-{total_ap_valid:.4f}",
        dirpath=Path(experiment_log_dir) / time_version / "contextual_tensorboard_logs" / version / "ap",
    )
    hits_k = config.get("hits_k")
    assert isinstance(hits_k, (list, tuple)), f"Hits@k have to be a Union[List, Tuple] of integers instead of {hits_k}"
    hits_checkpoint_callback_dict = {
        k: ModelCheckpoint(
            monitor=f"total_hits_{k}_valid",
            mode="max",
            verbose=True,
            save_last=True,
            save_top_k=3,
            filename=f"contextual-{k}" + "-{epoch:02d}-{total_hits_" + str(k) + "_valid:.4f}",
            dirpath=Path(experiment_log_dir) / time_version / "contextual_tensorboard_logs" / version / f"hits_{k}",
        )
        for k in hits_k
    }
    lr_monitor_callback = LearningRateMonitor(
        logging_interval="epoch",
        log_momentum=False,
    )

    loggers = {
        "tensorboard": tensorboard_logger,
        "tracking": tracking_logger,
    }

    callbacks = {
        "early_stopping": early_stopping,
        "loss_checkpoint": loss_checkpoint_callback,
        "accuracy_checkpoint": accuracy_checkpoint_callback,
        "auc_checkpoint": auc_checkpoint_callback,
        "f1_checkpoint": f1_checkpoint_callback,
        "ap_checkpoint": ap_checkpoint_callback,
        **hits_checkpoint_callback_dict,
        "lr_monitor": lr_monitor_callback,
    }

    if with_tune_report_callback:
        ray_variable_to_check = config.get("variable_to_check")
        filename = config.get("tune_report_checkpoint_filename")
        assert isinstance(filename, str)
        tune_report_checkpoint_callback = TuneReportCheckpointCallback(
            metrics=ray_variable_to_check,
            filename=filename,
            on="validation_end",
        )
        callbacks["tune_report_checkpoint"] = tune_report_checkpoint_callback

    return loggers, callbacks


def scheduler(conf: Dict[str, int]) -> FIFOScheduler:
    max_t = conf.get("max_epochs")
    assert isinstance(max_t, int)
    return ASHAScheduler(
        max_t=max_t,
        grace_period=1,
        reduction_factor=2,
    )


def search_alg(conf: Dict[str, Any]) -> Searcher:
    n_initial_points = conf.get("n_startup_trials")
    assert isinstance(n_initial_points, int)
    return HyperOptSearch(
        metric=conf.get("hyper_search_variable_to_check"),
        mode=conf.get("hyper_search_variable_to_check_mode"),
        n_initial_points=n_initial_points,
        random_state_seed=conf.get("random_seed"),
    )


def reporter(config: Dict[str, Any]) -> ProgressReporter:
    metric_columns = config.get("variable_to_check")
    assert isinstance(metric_columns, dict)
    return CLIReporter(
        parameter_columns=config.get("parameters_columns"),
        metric_columns=list(metric_columns) + ["training_iteration"],
    )
