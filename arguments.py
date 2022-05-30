import os
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from utils_common.class_registry import ClassRegistry

from latent_shift_predictor import regressor_registry
from models.gan_load import generator_registry
from deformators.load_deformator import deformator_registry

from utils_common.constants import ShiftDistribution
from deformators.load_deformator import shift_generator_registry


args = ClassRegistry()


@args.add_to_registry("exp")
@dataclass
class ExperimentArgs:
    config_dir: str = str(Path(__file__).resolve().parent / 'configs')
    config: str = "base_config.yaml"
    project: str = "GANLatent"
    name: str = MISSING
    seed: int = 1
    root: str = os.getenv("EXP_ROOT", ".")
    notes: str = "empty notes"
    logging: bool = True


@dataclass
class DeformationArgs:
    scale: float = 1.
    min_shift: float = 0.2
    distribution: ShiftDistribution = ShiftDistribution.UNIFORM


@args.add_to_registry("training")
@dataclass
class TrainingArgs:
    trainer: str = MISSING
    generator: str = MISSING
    deformator: str = MISSING
    regressor: str = MISSING
    shift_maker: str = MISSING
    directions_count: int = MISSING
    num_iters: Optional[int] = None
    batch_size: int = MISSING
    device: str = MISSING
    truncation: Optional[float] = None
    loss_funcs: List[str] = field(default_factory=list)
    loss_coefs: Optional[List[float]] = field(default_factory=list)


deformation = shift_generator_registry.make_dataclass_from_args('deformation')
args.add_to_registry("deformation")(deformation)


@dataclass
class OptimizerArgs:
    weight_decay: float = 0.
    lr: float = 1e-4
    betas: Tuple[float, ...] = (0.5, 0.999)


@args.add_to_registry("regressor_setup")
@dataclass
class RegressorArgs:
    model: str = MISSING
    loss_funcs: Optional[List[str]] = field(default_factory=list)
    loss_coefs: Optional[List[float]] = None
    optimizer: OptimizerArgs = OptimizerArgs




@args.add_to_registry("deformator_setup")
@dataclass
class DeformatorArgs:
    model: str = MISSING
    loss_funcs: Optional[List[str]] = field(default_factory=list)
    loss_coefs: Optional[List[float]] = None
    optimizer: OptimizerArgs = OptimizerArgs


@args.add_to_registry("logging")
@dataclass
class LoggingArgs:
    step_every: int = 10
    step_interpolation: int = 10000
    step_backup: int = 1000
    step_save: int = 10000
    step_losses: int = 20
    step_accuracy: int = 1000


@args.add_to_registry("checkpoint")
@dataclass
class CheckpointArgs:
    path: Optional[str] = None
    checkpointing_off: bool = False


Args = args.make_dataclass_from_classes("Args")


def load_config():
    conf_cli = OmegaConf.from_cli()

    if not conf_cli.get('exp', False):
        raise ValueError("No config")

    config_path = os.path.join(conf_cli.exp.config_dir, conf_cli.exp.config)
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(conf_file, conf_cli)

    return config


if __name__ == "__main__":
    config_base = OmegaConf.structured(Args)
    config_base_path = os.path.join(config_base.exp.config_dir, config_base.exp.config)

    conf_cli = OmegaConf.from_cli()

    if conf_cli.get('merge_with', False):
        print("merged with config: ", conf_cli['merge_with'])
        config_base = OmegaConf.merge(config_base, OmegaConf.load(conf_cli['merge_with']))
    
    if not Path(config_base_path).exists():
        with open(config_base_path, 'w') as fout:
            OmegaConf.save(config=config_base, f=fout.name)
