import abc
import wandb
import collections
import torch


class WandbLogger:
    def __init__(self, logging=False):
        self.logging = logging

    def setup(self, config, **kwargs):
        if not self.logging:
            return
        wandb.init(
            config=config,
            **kwargs
        )
            
    def finish(self):
        if not self.logging:
            return
        wandb.finish()

    def log(self, data):
        if not self.logging:
            return
        wandb.log(data)
    
    def log_images(self, images):
        if not self.logging:
            return

        for key, image, caption in images:
            wandb.log({
                key: wandb.Image(image, caption=caption)
            })
