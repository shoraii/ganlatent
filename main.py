import random
import torch
import arguments

from pathlib import Path
from omegaconf import OmegaConf
from trainer import trainer_registry


def run_experiment(model_config, exp_config):
    trainer = trainer_registry[exp_config.training.trainer](model_config, exp_config)
    trainer.setup()
    trainer.train()


def setup_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)


def setup_device(device):
    torch.cuda.set_device(device)


if __name__ == '__main__':
    config = arguments.load_config()
    setup_seed(config.exp.seed)
#     setup_device(config.training.device)

    models_config_path = str(Path(__file__).resolve().parent / 'configs/models.yaml')
    models_config = OmegaConf.load(models_config_path)

    run_experiment(models_config, config)
