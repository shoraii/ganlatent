from enum import Enum


HUMAN_ANNOTATION_FILE = 'human_annotation.txt'


class DeformatorType(Enum):
    FCA = 1
    LINEAR = 2
    ID = 3
    ORTHO = 4
    PROJECTIVE = 5
    RANDOM = 6
    FCB = 7
    MULTIFC = 8


class ShiftDistribution(Enum):
    NORMAL = 0
    UNIFORM = 1


DEFORMATOR_TYPE_DICT = {
    'fca': DeformatorType.FCA,
    'linear': DeformatorType.LINEAR,
    'id': DeformatorType.ID,
    'ortho': DeformatorType.ORTHO,
    'proj': DeformatorType.PROJECTIVE,
    'random': DeformatorType.RANDOM,
    'fcb': DeformatorType.FCB,
    'multifc': DeformatorType.MULTIFC
}


SHIFT_DISTRIDUTION_DICT = {
    'normal': ShiftDistribution.NORMAL,
    'uniform': ShiftDistribution.UNIFORM,
    None: None
}


WEIGHTS = {
    'BigGAN': 'models/pretrained/generators/BigGAN/G_ema.pth',
    'ProgGAN': 'models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth',
    'SN_MNIST': 'models/pretrained/generators/SN_MNIST',
    'SN_Anime': 'models/pretrained/generators/SN_Anime',
    'StyleGAN2': 'models/pretrained/StyleGAN2/stylegan2-car-config-f.pt',
    'StyleGAN2-e': 'models/pretrained/StyleGAN2/stylegan2-car-config-e.pt'
}
