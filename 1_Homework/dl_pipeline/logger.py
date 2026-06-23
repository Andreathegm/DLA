import wandb
from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf


class BaseLogger(ABC):
    ## Abstract base logger 

    @abstractmethod
    def log(self, metrics: dict, step: int = None) -> None:
        """Log a dictionary of scalar metrics at a given global step."""
        ...

    @abstractmethod
    def watch(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        """Hook into a model to log gradients / parameters periodically."""
        ...

    @abstractmethod
    def log_checkpoint(self, checkpoint_path: str, name: str) -> None:
        """Upload a checkpoint file as a versioned artifact."""
        ...

    @abstractmethod
    def finish(self) -> None:
        """Flush buffers and close the run cleanly."""
        ...


class WandBLogger(BaseLogger):
    """
    Thin wrapper around the WandB SDK.

    Exposes only what CustomTrainer needs, keeping the rest of the code
    completely decoupled from wandb internals.

    Args:
        cfg: The `logger` sub-config from your Hydra config
             (project, run_name, tags, mode, ...).
        full_cfg: The *entire* run config — passed as `config` to wandb.init
                  so every hyperparameter is stored on the run dashboard.

    Example config (configs/logger/wandb.yaml):
        project: dla-lab1
        run_name: null          # null → wandb auto-generates a name
        tags: []
        notes: ""
        mode: online            # online | offline | disabled
        watch_model: false
        watch_log: gradients    # gradients | parameters | all
        watch_log_freq: 100
        log_checkpoints: false  # upload .pt files as wandb Artifacts
    """

    def __init__(self, cfg: DictConfig, full_cfg: DictConfig = None):
        self.cfg = cfg
        self._log_checkpoints = cfg.get("log_checkpoints", False)

        # Serialize the entire experiment config so it shows up in the
        # wandb run dashboard under "Config" for easy comparison across runs.
        run_config = (
            OmegaConf.to_container(full_cfg, resolve=True)
            if full_cfg is not None
            else None
        )

        self.run = wandb.init(
            project=cfg.project,
            name=cfg.get("run_name", None),       # None → wandb picks a name
            tags=cfg.get("tags", []),
            notes=cfg.get("notes", ""),
            mode=cfg.get("mode", "online"),
            config=run_config,
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def log(self, metrics: dict, step: int = None) -> None:
        """
        Log scalar metrics. Mirrors the call signature used in CustomTrainer:
            self.logger.log({"train/loss": 0.42, ...}, step=self.global_step)
        """
        wandb.log(metrics, step=step)

    def watch(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        """
        Attach wandb hooks to a model so gradients/weights are streamed
        to the dashboard automatically.  Call this once, before training.
        """
        wandb.watch(model, log=log, log_freq=log_freq)

    def log_checkpoint(self, checkpoint_path: str, name: str) -> None:
        """
        Save a checkpoint .pt file as a wandb Artifact (versioned, resumable).
        Only runs when `log_checkpoints: true` in the logger config.
        """
        if not self._log_checkpoints:
            return
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        wandb.finish()

    # ------------------------------------------------------------------
    # Convenience helpers (not used by Trainer, but useful in train.py)
    # ------------------------------------------------------------------

    @property
    def run_name(self) -> str:
        return self.run.name

    @property
    def run_url(self) -> str:
        return self.run.url


class NullLogger(BaseLogger):
    """
    No-op logger — silently drops every call.

    Use it during unit tests or quick sanity checks where you don't want
    a wandb run to be created:

        logger = NullLogger()
        trainer = CustomTrainer(..., logger=logger, ...)
    """

    def log(self, metrics: dict, step: int = None) -> None:
        pass

    def watch(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        pass

    def log_checkpoint(self, checkpoint_path: str, name: str) -> None:
        pass

    def finish(self) -> None:
        pass


# ------------------------------------------------------------------
# Factory — used in train.py to keep construction logic in one place
# ------------------------------------------------------------------

def build_logger(cfg: DictConfig, full_cfg: DictConfig = None) -> BaseLogger:
    """
    Instantiate the right logger based on `cfg.logger.mode`.

    'disabled' → NullLogger  (no wandb process, zero overhead)
    anything else → WandBLogger

    Usage in train.py:
        logger = build_logger(cfg.logger, full_cfg=cfg)
    """
    mode = cfg.get("mode", "online")
    if mode == "disabled":
        return NullLogger()
    return WandBLogger(cfg, full_cfg=full_cfg)