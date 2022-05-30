from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.models import resnet18
from utils_common.class_registry import ClassRegistry


def save_hook(module, input, output):
    setattr(module, 'output', output)


regressor_registry = ClassRegistry()


@regressor_registry.add_to_registry("resnet_cls_magnitude_scalar")
class LatentShiftPredictor(nn.Module):
    def __init__(self, dim, downsample=None):
        super(LatentShiftPredictor, self).__init__()
        self.features_extractor = resnet18(pretrained=False)

        self.features_extractor.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, dim)
        self.shift_estimator = nn.Linear(512, 1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)

        self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()


@regressor_registry.add_to_registry("resnet_cls_magnitude_vector")
class LatentShiftPredictorVector(nn.Module):
    def __init__(self, dim, downsample=None):
        super(LatentShiftPredictorVector, self).__init__()
        self.features_extractor = resnet18(pretrained=False)

        self.features_extractor.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, dim)
        self.shift_estimator = nn.Linear(512, dim)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)

        self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift


class LeNetBackbone(nn.Module):
    def __init__(self, channels=3, width=2):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels * 2, 3 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(60 * width),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convnet(x)


class LeNetClassifierHead(nn.Module):
    def __init__(self, width, output_dim):
        super(LeNetClassifierHead, self).__init__()

        self.fc_logits = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, output_dim)
        )

    def forward(self, x):
        return self.fc_logits(x)


class LeNetShiftMagnitudeHead(nn.Module):
    def __init__(self, width, output_dim):
        super(LeNetShiftMagnitudeHead, self).__init__()
        self.fc_shift = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, output_dim)
        )

    def forward(self, x):
        return self.fc_shift(x)


@regressor_registry.add_to_registry("lenet_cls_magnitude_scalar")
class LeNetShiftPredictor(nn.Module):
    def __init__(self, dim, channels=3, width=2):
        super(LeNetShiftPredictor, self).__init__()

        self.backbone = LeNetBackbone(channels, width)
        self.fc_logits = LeNetClassifierHead(width, dim)
        self.fc_shift = LeNetShiftMagnitudeHead(width, 1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        features = self.backbone(torch.cat([x1, x2], dim=1))
        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()


@regressor_registry.add_to_registry("lenet_cls_magnitude_vector")
class LeNetShiftPredictorVector(nn.Module):
    def __init__(self, dim, channels=3, width=2):
        super(LeNetShiftPredictorVector, self).__init__()

        self.backbone = LeNetBackbone(channels, width)
        self.fc_logits = LeNetClassifierHead(width, dim)
        self.fc_shift = LeNetShiftMagnitudeHead(width, dim)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        features = self.backbone(torch.cat([x1, x2], dim=1))
        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift


@regressor_registry.add_to_registry("warped_gan_space")
class Reconstructor(nn.Module):
    def __init__(self, reconstructor_type, dim, channels=3):
        super(Reconstructor, self).__init__()
        self.reconstructor_type = reconstructor_type
        self.dim = dim
        self.channels = channels
        self.device = 'cuda:0' # TODO: AS ARGUMENT
        self.to(self.device)

        # === LeNet ===
        if self.reconstructor_type == 'LeNet':
            # Define LeNet backbone for feature extraction
            self.lenet_width = 2
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(self.channels * 2, 3 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(3 * self.lenet_width),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(3 * self.lenet_width, 8 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(8 * self.lenet_width),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(8 * self.lenet_width, 60 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(60 * self.lenet_width),
                nn.ReLU()
            )

            # Define classification head (for predicting warping functions (paths) indices)
            self.path_indices = nn.Sequential(
                nn.Linear(60 * self.lenet_width, 42 * self.lenet_width),
                nn.BatchNorm1d(42 * self.lenet_width),
                nn.ReLU(),
                nn.Linear(42 * self.lenet_width, self.dim)
            )

            # Define regression head (for predicting shift magnitudes)
            self.shift_magnitudes = nn.Sequential(
                nn.Linear(60 * self.lenet_width, 42 * self.lenet_width),
                nn.BatchNorm1d(42 * self.lenet_width),
                nn.ReLU(),
                nn.Linear(42 * self.lenet_width, 1)
            )

        # === ResNet ===
        elif self.reconstructor_type == 'ResNet':
            # Define ResNet18 backbone for feature extraction
            self.features_extractor = resnet18(pretrained=False)
            # Modify ResNet18 first conv layer so as to get 2 rgb images (concatenated as a 6-channel tensor)
            self.features_extractor.conv1 = nn.Conv2d(in_channels=6,
                                                      out_channels=64,
                                                      kernel_size=(7, 7),
                                                      stride=(2, 2),
                                                      padding=(3, 3), bias=False)
            nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.features = self.features_extractor.avgpool

            def save_hook(module, input, output):
                setattr(module, 'output', output)

            self.features.register_forward_hook(save_hook)

            # Define classification head (for predicting warping functions (paths) indices)
            self.path_indices = nn.Linear(512, self.dim)

            # Define regression head (for predicting shift magnitudes)
            self.shift_magnitudes = nn.Linear(512, 1)

    def forward(self, x1, x2):
        if self.reconstructor_type == 'LeNet':
            features = self.feature_extractor(torch.cat([x1, x2], dim=1))
            features = features.mean(dim=[-1, -2]).view(x1.shape[0], -1)
            return self.path_indices(features), self.shift_magnitudes(features).squeeze()
        elif self.reconstructor_type == 'ResNet':
            self.features_extractor(torch.cat([x1, x2], dim=1))
            features = self.features.output.view([x1.shape[0], -1])
            return self.path_indices(features), self.shift_magnitudes(features).squeeze()


@regressor_registry.add_to_registry("wgs_embed")
class EmbeddedReconstructor(nn.Module):
    def __init__(self, dim, channels=3):
        super(EmbeddedReconstructor, self).__init__()
        self.dim = dim
        self.channels = channels
        self.device = 'cuda:0' # TODO: AS ARGUMENT
        self.to(self.device)

        self.embed_gen = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=8,
                      kernel_size=(7, 7),
                      stride=(2, 2),
                      padding=(3, 3),
                      bias=False),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=(7, 7),
                      stride=(2, 2),
                      padding=(3, 3),
                      bias=False)
        )
        # Define ResNet18 backbone for feature extraction
        self.features_extractor = resnet18(pretrained=False)
        # Modify ResNet18 first conv layer so as to get 2 rgb images (concatenated as a 6-channel tensor)
        self.features_extractor.conv1 = nn.Conv2d(in_channels=32,
                                                  out_channels=64,
                                                  kernel_size=(7, 7),
                                                  stride=(2, 2),
                                                  padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.features = self.features_extractor.avgpool

        def save_hook(module, input, output):
            setattr(module, 'output', output)

        self.features.register_forward_hook(save_hook)

        # Define classification head (for predicting warping functions (paths) indices)
        self.path_indices = nn.Linear(512, self.dim)

        # Define regression head (for predicting shift magnitudes)
        self.shift_magnitudes = nn.Linear(512, 1)

    def forward(self, x1, x2):
        x1e = self.embed_gen(x1)
        x2e = self.embed_gen(x2)
        self.features_extractor(torch.cat([x1e, x2e], dim=1))
        features = self.features.output.view([x1.shape[0], -1])
        return self.path_indices(features), self.shift_magnitudes(features).squeeze()


@regressor_registry.add_to_registry("pm_wgs")
class PlusMinusReconstructor(nn.Module):
    def __init__(self, dim, channels=3):
        super(PlusMinusReconstructor, self).__init__()
        self.dim = dim
        self.channels = channels
        self.device = 'cuda:0' # TODO: AS ARGUMENT
        self.to(self.device)
        # Define ResNet18 backbone for feature extraction
        self.features_extractor = resnet18(pretrained=False)
        # Modify ResNet18 first conv layer so as to get 2 rgb images (concatenated as a 6-channel tensor)
        self.features_extractor.conv1 = nn.Conv2d(in_channels=9,
                                                  out_channels=64,
                                                  kernel_size=(7, 7),
                                                  stride=(2, 2),
                                                  padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.features = self.features_extractor.avgpool

        def save_hook(module, input, output):
            setattr(module, 'output', output)

        self.features.register_forward_hook(save_hook)

        # Define classification head (for predicting warping functions (paths) indices)
        self.path_indices = nn.Linear(512, self.dim)

        # Define regression head (for predicting shift magnitudes)
        self.shift_magnitudes = nn.Linear(512, 1)

    def forward(self, x_minus, x, x_plus):
        self.features_extractor(torch.cat([x_minus, x, x_plus], dim=1))
        features = self.features.output.view([x.shape[0], -1])
        return self.path_indices(features), self.shift_magnitudes(features).squeeze()


@regressor_registry.add_to_registry("pm_wgs_embed")
class PlusMinusEmbedReconstructor(nn.Module):
    def __init__(self, dim, channels=3):
        super(PlusMinusEmbedReconstructor, self).__init__()
        self.dim = dim
        self.channels = channels
        self.device = 'cuda:0' # TODO: AS ARGUMENT
        self.to(self.device)
        self.embed_gen = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=8,
                      kernel_size=(7, 7),
                      stride=(2, 2),
                      padding=(3, 3),
                      bias=False),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=(7, 7),
                      stride=(2, 2),
                      padding=(3, 3),
                      bias=False)
        )
        # Define ResNet18 backbone for feature extraction
        self.features_extractor = resnet18(pretrained=False)
        # Modify ResNet18 first conv layer so as to get 2 rgb images (concatenated as a 6-channel tensor)
        self.features_extractor.conv1 = nn.Conv2d(in_channels=16 * 3,
                                                  out_channels=64,
                                                  kernel_size=(7, 7),
                                                  stride=(2, 2),
                                                  padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.features = self.features_extractor.avgpool

        def save_hook(module, input, output):
            setattr(module, 'output', output)

        self.features.register_forward_hook(save_hook)

        # Define classification head (for predicting warping functions (paths) indices)
        self.path_indices = nn.Linear(512, self.dim)

        # Define regression head (for predicting shift magnitudes)
        self.shift_magnitudes = nn.Linear(512, 1)

    def forward(self, x_minus, x, x_plus):
        x_minus = self.embed_gen(x_minus)
        x = self.embed_gen(x)
        x_plus = self.embed_gen(x_plus)
        self.features_extractor(torch.cat([x_minus, x, x_plus], dim=1))
        features = self.features.output.view([x.shape[0], -1])
        return self.path_indices(features), self.shift_magnitudes(features).squeeze()
