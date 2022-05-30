import os
import sys
import json
import wandb
import omegaconf

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from utils import make_noise, is_conditional
from omegaconf import OmegaConf

from utils_common.train_log import StreamingMeans
from utils_common.constants import DeformatorType, ShiftDistribution
from utils_common.visualization import make_interpolation_chart, fig_to_image
from utils_common.loggers import WandbLogger
from utils_common.visualization import inspect_all_directions
from utils_common.class_registry import ClassRegistry
from utils import is_conditional

from latent_shift_predictor import regressor_registry
from models.gan_load import generator_registry
from deformators.load_deformator import deformator_registry, shift_generator_registry
from netdissect.nethook import InstrumentedModel
from losses import LatentDirectionSearchLoss


trainer_registry = ClassRegistry()


@trainer_registry.add_to_registry(name='base_trainer')
class Trainer:
    def __init__(self, models_config, config):        
        
        config['deformator_args'] = models_config['deformators'][config.training.deformator]
        config['generator_args'] = models_config['generators'][config.training.generator]
        config['regressor_args'] = models_config['regressors'][config.training.regressor]
        
        self.config = config
                
        # fixed latent noise to include into interpolation charts
        self.fixed_test_noise = None

    def setup(self):
        self._setup_device()
        self._setup_experiment_dir()
        
        self._setup_generator()
        self._setup_regressor()
        self._setup_deformator()
        self._to_device(self.device)
        
        self._setup_optimizers()
        self._setup_losses()
        self._setup_logger()

    def _setup_device(self):
        config_device = self.config.training['device'].lower()

        if config_device == 'cpu':
            device = 'cpu'
        elif config_device.isdigit():
            device = 'cuda:{}'.format(config_device)
        elif config_device.startswith('cuda:'):
            device = config_device
        else:
            raise ValueError("Incorrect Device Type")

        try:
            torch.randn(1).to(device)
            print("Device: {}".format(device))
        except Exception as e:
            print("Could not use device {}, {}".format(device, e))
            print("Set device to CPU")
            device = 'cpu'
        
        self.device = torch.device(device)

    def _setup_experiment_dir(self):
        base_root = Path(__file__).resolve().parent
        num = 0
        exp_dir = "{}_{}".format(self.config.exp.name, str(num).zfill(3))

        exp_path = base_root / exp_dir
        while True:
            if exp_path.exists():
                num += 1
                exp_dir = "{}_{}".format(self.config.exp.name, str(num).zfill(3))
                print(exp_path, "already exists: move to", exp_dir)
            else:
                break
            exp_path = base_root / exp_dir
        self.experiment_dir = str(exp_path)
        os.makedirs(self.experiment_dir)

        with open(os.path.join(self.experiment_dir, 'config.yaml'), 'w') as f:
            omegaconf.OmegaConf.save(config=self.config, f=f.name)

        with open(os.path.join(self.experiment_dir, 'run_command.sh'), 'w') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')

        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.mkdir(self.checkpoint_dir)
        self.models_dir = os.path.join(self.experiment_dir, 'models')
        os.mkdir(self.models_dir)

    # TODO: later
    @classmethod
    def from_experiment(cls, exp_dir):
        return Trainer()

    def _setup_generator(self):
        generator = generator_registry[self.config.training.generator](
            **self.config.generator_args
        ).to(self.device)
        self.generator = InstrumentedModel(generator)

    def _setup_regressor(self):
        self.regressor = regressor_registry[self.config.training.regressor](
            **self.config.regressor_args
        ).to(self.device)

    def _setup_deformator(self):
        self.deformator = deformator_registry[self.config.training.deformator](
            **self.config.deformator_args
        )(
            self.generator
        ).to(self.device)

        self.shift_maker = shift_generator_registry[self.config.training.shift_maker](
            **self.config.deformation[self.config.training.shift_maker]
        )(
            self.generator
        )

        self.deformator.shift_maker = self.shift_maker

    def _setup_logger(self):
        self.logger = WandbLogger(self.config.exp.logging)

        self.logger.setup(
            config=self.config,
            project=self.config.exp.project,
            name=self.config.exp.name,
            notes=self.config.exp.notes
        )

    def _setup_optimizers(self):
        self.deform_optimizer = torch.optim.Adam(
            self.deformator.parameters(),
            **self.config.deformator_setup.optimizer
        )

        self.regressor_optimizer = torch.optim.Adam(
            self.regressor.parameters(),
            **self.config.regressor_setup.optimizer
        )

    def _setup_losses(self):
        self.combined_loss = LatentDirectionSearchLoss(
            self.config.training.loss_funcs, self.config.training.loss_coefs
        )

    def calc_loss(self, z, z_edited,
                  target_indices, target_shifts,
                  iter_info,
                  no_grad=False, log_event="train_loss"):

        with torch.no_grad() if no_grad else torch.enable_grad():
            if is_conditional(self.generator):
                classes = torch.from_numpy(np.random.choice(self.generator.model.target_classes.cpu()).repeat(self.config.training.batch_size)).to(self.device)
                imgs = self.generator(z, classes)
                imgs_shifted = self.generator(z_edited, classes)
            else:
                imgs = self.generator(z)
                imgs_shifted = self.generator(z_edited)
             
            logits, shift_prediction = self.regressor(imgs, imgs_shifted)

        losses = self.combined_loss(
            logits,
            shift_prediction,
            target_indices,
            target_shifts,
            z,
            z_edited,
            self.generator
        )

        # TODO: additional losses
        # deform_losses = self.combined_loss_deform(
        #     logits,
        #     shift_prediction,
        #     target_indices,
        #     target_shifts,
        #     self.generator
        # )

        iter_info.update({f"{log_event}/{k}": v for k, v in losses.items()})
        return losses["total"]

    def _to_device(self, device):
        self.generator.to(device)
        self.deformator.to(device)
        self.deformator.shift_maker.to(device)
        self.regressor.to(device)

    def to_train(self):
        self.deformator.train()
        self.regressor.train()

    def to_eval(self):
        self.deformator.eval()
        self.regressor.eval()

    def zero_grad(self):
        self.deform_optimizer.zero_grad()
        self.regressor_optimizer.zero_grad()

    def step(self):
        self.deform_optimizer.step()
        self.regressor_optimizer.step()

    def start_from_checkpoint(self):
        step = 0
        if self.config.checkpoint.path:
            state_dict = torch.load(self.config.checkpoint.path)
            step = state_dict['step']
            self.deformator.load_state_dict(state_dict['deformator'])
            self.regressor.load_state_dict(state_dict['regressor'])
            self.regressor_optimizer.load_state_dict(state_dict['regressor_optimizer'])
            self.deform_optimizer.load_state_dict(state_dict['deformator_optimizer'])
            print('starting from step {}'.format(step))
        return step

    def log(self, iter_info):
        step = self.current_step + 1
                
        if step % self.config.logging.step_losses == 0:
            self.logger.log(iter_info.to_dict())

        if step % self.config.logging.step_interpolation == 0:
            self.log_interpolation()

        if step % self.config.logging.step_backup == 0:
            self.make_checkpoint()

        if step % self.config.logging.step_accuracy == 0:
            self.log_accuracy()

        if step % self.config.logging.step_save == 0:
            self.save_models()

    def log_interpolation(self):
        noise = make_noise(
            1, self.generator.model.dim_z, self.config.training.truncation
        ).to(self.device)

        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
                
        directions_per_image = 15
        dir_count = self.config.training.directions_count

        for start in range(0, dir_count - 1, directions_per_image):
            dims = range(start, min(start + directions_per_image, dir_count))
            for z, prefix in zip([noise, self.fixed_test_noise], ['rand', 'fixed']):
                fig = make_interpolation_chart(
                    self.generator, deformator=self.deformator, z=z,
                    shifts_count=3, directions=dims, shifts_r=self.shift_maker.shift_scale,
                    dpi=250, figsize=(int(12), int(0.5 * len(dims)) + 2))

                fig.canvas.draw()
                plt.close(fig)
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
                img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
                self.logger.log({
                    "interpolation/{}/{}_{}".format(prefix, dims[0], dims[-1]): wandb.Image(img)
                })

    def make_checkpoint(self):
        if self.config.checkpoint.checkpointing_off:
            return

        ckpt = self.get_checkpoint()
        torch.save(ckpt, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def get_checkpoint(self):
        state_dict = {
            'step': self.current_step,
            'deformator': self.deformator.state_dict(),
            'regressor': self.regressor.state_dict(),
            'deformator_optimizer': self.deform_optimizer.state_dict(),
            'regressor_optimizer': self.regressor_optimizer.state_dict()
        }
        return state_dict

    def save_models(self):
        torch.save(self.deformator.state_dict(),
                   os.path.join(self.models_dir, 'deformator_{}.pt'.format(self.current_step)))
        torch.save(self.regressor.state_dict(),
                   os.path.join(self.models_dir, 'regressor_{}.pt'.format(self.current_step)))

    def log_accuracy(self):
        self.deformator.eval()
        self.regressor.eval()

        accuracy = self.validate_regressor()
        self.logger.log({
            "metrics/accuracy_regressor": accuracy.item()
        })

        self.deformator.train()
        self.regressor.train()

    def train(self):
        self.to_train()

        # TODO: to take into account conditional GAN
        # should_gen_classes = is_conditional(G)

        recovered_step = self.start_from_checkpoint()
        iter_info = StreamingMeans()

        for self.current_step in range(recovered_step, self.config.training.num_iters, 1):
            self.train_step(iter_info)
            if (self.current_step + 1) % self.config.logging.step_every == 0:
                self.log(iter_info)

        self.save_results_charts()
        self.logger.finish()

    def train_step(self, iter_info):
        self.zero_grad()

        z = make_noise(
            self.config.training.batch_size,
            self.generator.model.dim_z,
            self.config.training.truncation
        ).to(self.device)

        # input dim is coded into deformator
        z_edited, target_indices, shifts = self.deformator(z)

        loss = self.calc_loss(z, z_edited, target_indices, shifts, iter_info)
        loss.backward()
        self.step()

    @torch.no_grad()
    def save_results_charts(self):
        self.deformator.eval()
        self.generator.eval()

        z = make_noise(
            3,
            self.generator.model.dim_z,
            self.config.training.truncation
        ).cuda()

        shift_scale = int(self.config.training.deformation.shift_scale)

        inspect_all_directions(
            self.generator,
            self.deformator,
            os.path.join(self.experiment_dir, 'charts_s{}'.format(shift_scale)),
            zs=z,
            shifts_r=float(shift_scale))

        inspect_all_directions(
            self.generator,
            self.deformator,
            os.path.join(self.experiment_dir, 'charts_s{}'.format(int(3 * shift_scale))),
            zs=z,
            shifts_r=3 * float(shift_scale))

    @torch.no_grad()
    def validate_regressor(self):
        n_steps = 100

        if is_conditional(self.generator):
            classes = torch.from_numpy(np.random.choice(self.generator.model.target_classes.cpu()).repeat(self.config.training.batch_size)).to(self.device)
        percents = torch.empty([n_steps])
        for step in range(n_steps):
            z = make_noise(
                self.config.training.batch_size,
                self.generator.model.dim_z,
                self.config.training.truncation
            ).cuda()

            z_edited, target_indices, shifts = self.deformator(z)

            if is_conditional(self.generator):
                imgs = self.generator(z, classes)
                imgs_edited = self.generator(z_edited, classes)
            else:
                imgs = self.generator(z)
                imgs_edited = self.generator(z_edited)

            logits, _ = self.regressor(imgs, imgs_edited)
            percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

        return percents.mean()
