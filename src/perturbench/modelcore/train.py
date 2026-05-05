import logging
from typing import List
from pathlib import Path
import hydra
import lightning as L
import torch
import pandas as pd
from omegaconf import DictConfig, open_dict
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from perturbench.modelcore.utils import multi_instantiate
from perturbench.modelcore.models import PerturbationModel
from hydra.core.hydra_config import HydraConfig


log = logging.getLogger(__name__)


class RollingCheckpoint(Callback):
    """Overwrite one checkpoint every N epochs, at train end, and on interruption."""

    def __init__(
        self,
        dirpath: str | Path,
        filename: str = "last.ckpt",
        every_n_epochs: int = 50,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.every_n_epochs = every_n_epochs

    def _save(self, trainer: L.Trainer) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / self.filename
        log.info("Saving rolling checkpoint to %s", path)
        trainer.save_checkpoint(path)

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self.every_n_epochs <= 0:
            return
        epoch_number = trainer.current_epoch + 1
        if epoch_number % self.every_n_epochs == 0:
            self._save(trainer)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._save(trainer)

    def on_exception(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        exception: BaseException,
    ) -> None:
        self._save(trainer)


def train(runtime_context: dict):
    cfg = runtime_context["cfg"]

    # Enable TensorCore-friendly FP32 matmul precision. Under precision=16 the
    # main matmuls are FP16 already; this only affects residual FP32 paths
    # (optimizer updates, loss reductions, log math). Accepted deviation from
    # "highest" is well below the across-fold noise floor.
    torch.set_float32_matmul_precision("high")

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        seed=cfg.seed,
    )

    log.info("Instantiating model <%s>", cfg.model._target_)
    model: PerturbationModel = hydra.utils.instantiate(cfg.model, datamodule=datamodule)

    # Allow overriding the LR scheduler / monitor key after instantiation so we
    # can switch from val_loss to train_loss when val is disabled, without
    # needing to plumb the kwarg through every model subclass.
    lr_monitor_override = cfg.get("lr_monitor_key")
    if lr_monitor_override is not None:
        model.lr_monitor_key = lr_monitor_override
        log.info("Overriding model.lr_monitor_key -> %s", lr_monitor_override)

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = multi_instantiate(cfg.get("callbacks"))
    rolling_cfg = cfg.get("rolling_checkpoint") or cfg.get("rolling_latest_checkpoint")
    if rolling_cfg is not None:
        checkpoint_dir = rolling_cfg.get("dirpath")
        if checkpoint_dir is None:
            checkpoint_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
        callbacks.append(
            RollingCheckpoint(
                dirpath=checkpoint_dir,
                filename=rolling_cfg.get("filename", "last.ckpt"),
                every_n_epochs=rolling_cfg.get("every_n_epochs", 50),
            )
        )

    log.info("Instantiating loggers...")
    loggers: List[Logger] = multi_instantiate(cfg.get("logger"))

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    summary_metrics_dict = {}
    if cfg.get("test"):
        log.info("Starting testing!")
        original_eval_save_dir = Path(model.evaluation_config.save_dir)

        def resolve_test_ckpt(test_ckpt_path):
            if test_ckpt_path in (None, "none", "current"):
                return None
            if test_ckpt_path == "best":
                if (
                    trainer.checkpoint_callback is None
                    or trainer.checkpoint_callback.best_model_path == ""
                ):
                    return None
                return "best"
            return test_ckpt_path

        if cfg.get("train") and cfg.get("evaluate_train_checkpoints"):
            final_ckpt = original_eval_save_dir.parent / "checkpoints" / "last.ckpt"
            evaluations = [
                ("best_train_loss", resolve_test_ckpt("best")),
                ("final", str(final_ckpt) if final_ckpt.exists() else None),
            ]
        elif cfg.get("train"):
            evaluations = [("summary", resolve_test_ckpt(cfg.get("test_ckpt_path", "best")))]
        else:
            evaluations = [("summary", cfg.get("ckpt_path"))]

        summary_tables = []
        for label, ckpt_path in evaluations:
            eval_save_dir = original_eval_save_dir / label
            with open_dict(model.evaluation_config):
                model.evaluation_config.save_dir = str(eval_save_dir)
            log.info("Testing %s checkpoint with ckpt_path=%s", label, ckpt_path)
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

            if model.summary_metrics is not None:
                value_col = model.summary_metrics.columns[0]
                summary = model.summary_metrics.rename(columns={value_col: label})
                summary_tables.append(summary)
                for metric, value in summary[label].to_dict().items():
                    summary_metrics_dict[f"{label}_{metric}"] = value

        with open_dict(model.evaluation_config):
            model.evaluation_config.save_dir = str(original_eval_save_dir)

        if summary_tables:
            combined_summary = pd.concat(summary_tables, axis=1)
            original_eval_save_dir.mkdir(parents=True, exist_ok=True)
            combined_summary.to_csv(
                original_eval_save_dir / "summary.csv",
                index_label="metric",
            )

    test_metrics = trainer.callback_metrics
    # merge train and test metrics, converting tensors to plain floats
    # so the dict is picklable (required by joblib multirun launcher)
    metric_dict = {}
    for d in (train_metrics, test_metrics, summary_metrics_dict):
        for k, v in d.items():
            metric_dict[k] = v.item() if hasattr(v, "item") else v

    return metric_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    runtime_context = {"cfg": cfg, "trial_number": HydraConfig.get().job.get("num")}

    ## Train the model
    global metric_dict
    metric_dict = train(runtime_context)

    ## Combined metric
    metrics_use = cfg.get("metrics_to_optimize")
    if metrics_use:
        combined_metric = sum(
            [metric_dict.get(metric) * weight for metric, weight in metrics_use.items()]
        )
        return combined_metric


if __name__ == "__main__":
    main()
