import typing as tp

import torch
import numpy as np

from deformators.deformators import LatentDeformator, ActivationVectorDeformator, WarpedDeformator
from deformators.deformators import Randomizer
from utils_common.constants import DeformatorType, DEFORMATOR_TYPE_DICT, SHIFT_DISTRIDUTION_DICT
from utils_common.class_registry import ClassRegistry
from utils import is_conditional

deformator_registry = ClassRegistry()
shift_generator_registry = ClassRegistry()


@shift_generator_registry.add_to_registry("randomizer")
def build_shift_maker_factory(
    directions_count,
    shift_scale,
    min_shift,
    shift_distribution
):
    def build_randomizer(instrumented_generator):
        distr = SHIFT_DISTRIDUTION_DICT[shift_distribution.lower()]

        return Randomizer(
            directions_count=directions_count,
            latent_dim=np.prod(instrumented_generator.model.dim_z),
            shift_scale=shift_scale,
            min_shift=min_shift,
            shift_distribution=distr
        )
    return build_randomizer


# not used yet
class DeformatorWrapper:
    def __init__(self, base_deformator):
        self.deformator = base_deformator


@deformator_registry.add_to_registry("latent_vector")
def make_latent_deformator(
    directions_count: int,
    inner_dim: int = 1024,
    type: DeformatorType = DeformatorType.FC,
    random_init: bool = False,
    bias: bool = True
):
    deformator_type = DEFORMATOR_TYPE_DICT[type.lower()]

    def build_latent_deformator(instrumented_generator):
        return LatentDeformator(
            shift_dim=instrumented_generator.model.dim_z,
            input_dim=directions_count,
            inner_dim=inner_dim,
            type=deformator_type,
            random_init=random_init,
            bias=bias
        )

    return build_latent_deformator


@deformator_registry.add_to_registry("activations_vector")
def activation_vector_deformator(
    directions_count: int,
    layers: tp.List[str]
):
    def build_activation_deformator(instrumented_generator):
        instrumented_generator.retain_layers(layers)
        dim_z = instrumented_generator.model.dim_z
        z = torch.randn(1, dim_z)

        if is_conditional(instrumented_generator.model):
            cl_emb = instrumented_generator.model.shared([239, ])
            instrumented_generator(z, cl_emb)
        else:
            instrumented_generator(z)

        shift_dims = []

        for key, features in instrumented_generator.retained_features().items():
            shift_dims.append(features.size()[1:])

        return ActivationVectorDeformator(
            instrumented_generator=instrumented_generator,
            shift_dims=shift_dims,
            input_dim=directions_count,
            layers=layers
        )
    return build_activation_deformator


@deformator_registry.add_to_registry("warped_gan_space")
def activation_vector_deformator(
    shift_dim: int,
    num_support_dipoles: int,
    support_vectors_dim: int,
    learn_alphas: bool = False,
    learn_gammas: bool = False,
    gamma: float = None, 
    min_shift_magnitude: float = 0.2,
    max_shift_magnitude: float = 0.5
):
    def build_warped_deformator(g):
        return WarpedDeformator(
            shift_dim=shift_dim,
            num_support_dipoles=num_support_dipoles,
            support_vectors_dim=support_vectors_dim,
            learn_alphas=learn_alphas,
            learn_gammas=learn_gammas,
            gamma=gamma,
            min_shift_magnitude=min_shift_magnitude,
            max_shift_magnitude=max_shift_magnitude
        )
    return build_warped_deformator


@deformator_registry.add_to_registry("latent_activation_vector")
def make_latent_activation_deformator(generator):
    ...

