from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar
from omegaconf import OmegaConf, MISSING
import hydra
from hydra.utils import instantiate

T = TypeVar('T')

@dataclass
class BaseConfig:
    """Base configuration class with common settings"""
    # Experiment metadata
    experiment_name: str = MISSING
    run_name: Optional[str] = None
    seed: int = 1
    deterministic: bool = True
    
    # Device configuration
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Precision settings
    precision: str = "32-true"
    mixed_precision: bool = False
    
    # Debugging
    debug: bool = False
    fast_dev_run: bool = False

@dataclass
class ModelConfig(BaseConfig):
    """Model-specific configuration"""
    model_name: str = MISSING
    model_params: Dict[str, Any] = MISSING
    
    # Loss function configuration
    loss_fn: str = "cross_entropy"
    loss_params: Dict[str, Any] = None
    
    # Optimizer configuration
    optimizer: str = "adam"
    optimizer_params: Dict[str, Any] = None
    
    # Learning rate scheduler
    scheduler: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = None

@dataclass 
class DataConfig(BaseConfig):
    """Data pipeline configuration"""
    dataset_name: str = MISSING
    data_dir: str = "./data"
    batch_size: int = 32
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Augmentation settings
    augment: bool = True
    augment_params: Dict[str, Any] = None
    
    # Normalization
    normalize: bool = True
    mean: list = None
    std: list = None

@dataclass
class TrainingConfig(BaseConfig):
    """Training loop configuration"""
    max_epochs: int = 100
    min_epochs: int = 1
    max_steps: Optional[int] = None
    gradient_clip_val: Optional[float] = None
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    monitor: str = "val_loss"
    mode: str = "min"
    
    # Checkpointing
    checkpoint: bool = True
    checkpoint_dir: str = "./checkpoints"
    save_top_k: int = 3
    every_n_epochs: int = 1

@dataclass
class LoggingConfig(BaseConfig):
    """Logging and tracking configuration"""
    logger: str = "wandb"
    project_name: str = "deep-learning-experiments"
    entity: Optional[str] = None
    log_model: bool = True
    log_gradients: bool = False
    
    # Visualization
    save_figures: bool = True
    figure_dir: str = "./figures"
    
    # Frequency
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0

@dataclass
class ExperimentConfig:

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    logging: LoggingConfig
    
    # Hydra-specific
    defaults: list = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        return OmegaConf.structured(cls)
    
    def merge_with(self, other: Dict[str, Any]) -> 'ExperimentConfig':
        """Merge with dictionary overrides"""
        return OmegaConf.merge(self, other)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return OmegaConf.to_container(self)