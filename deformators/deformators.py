import typing as tp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_common.constants import DeformatorType, ShiftDistribution
from utils_common.ortho_utils import torch_expm
from utils_common.class_registry import ClassRegistry

deformator_registry = ClassRegistry()


class Randomizer:
    def __init__(self,
                 directions_count,
                 latent_dim,
                 shift_scale,
                 min_shift,
                 shift_distribution):

        self.latent_dim = latent_dim
        self.directions_count = directions_count
        self.shift_scale = shift_scale
        self.min_shift = min_shift
        self.distribution = ShiftDistribution.UNIFORM \
            if shift_distribution == 'UNIFOFM' else ShiftDistribution.NORMAL
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device

    def __call__(self, z):
        batch_size = z.size(0)
        target_indices = torch.randint(
            0, self.directions_count, [batch_size], device=self.device)

        if self.distribution == ShiftDistribution.NORMAL:
            shifts = torch.randn(target_indices.shape, device=self.device)
        elif self.distribution == ShiftDistribution.UNIFORM:
            shifts = 2.0 * torch.rand(target_indices.shape, device=self.device) - 1.0
        else:
            raise TypeError("Incorrect ShiftDistribution")

        shifts = self.shift_scale * shifts
        shifts[(shifts < self.min_shift) & (shifts > 0)] = self.min_shift
        shifts[(shifts > -self.min_shift) & (shifts < 0)] = -self.min_shift

        z_shift = torch.zeros([batch_size, self.directions_count], device=self.device)
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift


class Split(nn.Module):
    def __init__(self, z_dim: int, *modules: torch.nn.Module):
        super().__init__()
        self.splits = nn.ModuleList(modules)
        self.z_dim = z_dim

    def forward(self, indices, z):
        b_size = len(indices)
        output = torch.zeros((b_size, self.z_dim), dtype=torch.float, device='cuda')
        for i, index in enumerate(indices):
            output[i] = self.splits[int(index)](z.unsqueeze(0))
        return output

class FCDeformatorBlock(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(out_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.act1 = nn.ELU()

        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act2 = nn.ELU()

        self.fc3 = nn.Linear(out_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.act3 = nn.ELU()

        self.fc4 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x1 = self.fc1(input)
        x = self.act1(self.bn1(x1))

        x2 = self.fc2(x)
        x = self.act2(self.bn2(x2 + x1))

        x3 = self.fc3(x)
        x = self.act3(self.bn3(x3 + x2 + x1))

        x = self.fc4(x)
        return x

class LatentDeformator(nn.Module):
    def __init__(self, shift_dim, input_dim=None, inner_dim=64,
                 type=DeformatorType.FCA, random_init=False, bias=True):
        super().__init__()
        self.type = type
        self.shift_dim = shift_dim
        self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
        self.out_dim = np.prod(self.shift_dim)
        self.shift_maker = None
        self.inner_dim = inner_dim
        self.directions_count = 32

        if self.type == DeformatorType.FCA:
            self.fc1 = nn.Linear(self.input_dim, inner_dim)
            self.bn1 = nn.BatchNorm1d(inner_dim)
            self.act1 = nn.ELU()

            self.fc2 = nn.Linear(inner_dim, inner_dim)
            self.bn2 = nn.BatchNorm1d(inner_dim)
            self.act2 = nn.ELU()

            self.fc3 = nn.Linear(inner_dim, inner_dim)
            self.bn3 = nn.BatchNorm1d(inner_dim)
            self.act3 = nn.ELU()

            self.fc4 = nn.Linear(inner_dim, self.out_dim)
            self.fc4s = nn.Linear(inner_dim, 1)
        elif self.type == DeformatorType.FCB:
            self.seeder = nn.Embedding(self.directions_count, self.directions_count)
            self.fc1 = nn.Linear(self.directions_count, inner_dim)
            self.bn1 = nn.BatchNorm1d(inner_dim)
            self.act1 = nn.ELU()

            self.fc2 = nn.Linear(inner_dim, inner_dim)
            self.bn2 = nn.BatchNorm1d(inner_dim)
            self.act2 = nn.ELU()

            self.fc3 = nn.Linear(inner_dim, inner_dim)
            self.bn3 = nn.BatchNorm1d(inner_dim)
            self.act3 = nn.ELU()

            self.fc4 = nn.Linear(inner_dim, self.out_dim)
            self.fc4s = nn.Linear(inner_dim, 1)
        elif self.type == DeformatorType.MULTIFC:
            self.splits = []
            self.directions_count = 32
            for i in range(self.directions_count):
                self.splits.append(
                    FCDeformatorBlock(self.out_dim)
                )
            self.split_def = Split(self.out_dim, *self.splits)
            self.bn = nn.BatchNorm1d(self.out_dim)
            self.act = nn.ReLU()
            self.fc = nn.Linear(self.out_dim, self.out_dim, bias=False)
        elif self.type in [DeformatorType.LINEAR, DeformatorType.PROJECTIVE]:
            self.linear = nn.Linear(self.input_dim, self.out_dim, bias=bias)
            self.linear.weight.data = torch.zeros_like(self.linear.weight.data)

            min_dim = int(min(self.input_dim, self.out_dim))
            self.linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
            if random_init:
                self.linear.weight.data = 0.1 * torch.randn_like(self.linear.weight.data)

        elif self.type == DeformatorType.ORTHO:
            assert self.input_dim == self.out_dim, 'In/out dims must be equal for ortho'
            self.log_mat_half = nn.Parameter((1.0 if random_init else 0.001) * torch.randn(
                [self.input_dim, self.input_dim]), True)

        elif self.type == DeformatorType.RANDOM:
            self.linear = torch.empty([self.out_dim, self.input_dim])
            nn.init.orthogonal_(self.linear)

    def forward(self, z, basis_shift=None):
        if basis_shift is None:
            target_indices, shifts, basis_shift = self.shift_maker(z)
        else:
            target_indices = basis_shift.abs().nonzero()[:, 1]
            shifts = torch.tensor([basis_shift[idx, ind.item()] for idx, ind in enumerate(target_indices)])

        if self.type == DeformatorType.ID:
            return z

        input = basis_shift.view([-1, self.input_dim])
        shifts = shifts.to('cuda')
        if len(target_indices) == 0:
            return z, target_indices, shifts
        if self.type == DeformatorType.FCA:
            x1 = self.fc1(input)
            x = self.act1(self.bn1(x1))

            x2 = self.fc2(x)
            x = self.act2(self.bn2(x2 + x1))

            x3 = self.fc3(x)
            x = self.act3(self.bn3(x3 + x2 + x1))

            s = self.fc4s(x)
            x = self.fc4(x)

            x = F.normalize(x, eps=5e-4)

            shifts = shifts.view(-1, 1)

            s = s.view(-1, 1)
            s = torch.exp(s)

            x = shifts * s * x

            out = x + z
        elif self.type == DeformatorType.FCB:
            inp = target_indices.view(-1, 1)
            x = self.seeder(inp).squeeze(1)
            x1 = self.fc1(x)
            x = self.act1(self.bn1(x1))

            x2 = self.fc2(x)
            x = self.act2(self.bn2(x2 + x1))

            x3 = self.fc3(x)
            x = self.act3(self.bn3(x3 + x2 + x1))

            s = self.fc4s(x)
            x = self.fc4(x)

            x = F.normalize(x, eps=5e-4)

            shifts = shifts.view(-1, 1)

            s = s.view(-1, 1)
            s = torch.exp(s)

            x = shifts * s * x

            out = x + z
        elif self.type == DeformatorType.MULTIFC:
            x = self.split_def(target_indices, z)
            input_norm = torch.norm(input, dim=1, keepdim=True)
            out = (input_norm / torch.norm(x, dim=1, keepdim=True)) * x + z
        elif self.type == DeformatorType.LINEAR:
            out = self.linear(input) + z
        elif self.type == DeformatorType.PROJECTIVE:
            input_norm = torch.norm(input, dim=1, keepdim=True)
            out = self.linear(input)
            out = (input_norm / torch.norm(out, dim=1, keepdim=True)) * out + z
        elif self.type == DeformatorType.ORTHO:
            mat = torch_expm((self.log_mat_half - self.log_mat_half.transpose(0, 1)).unsqueeze(0))
            out = F.linear(input, mat) + z
        elif self.type == DeformatorType.RANDOM:
            self.linear = self.linear.to(input.device)
            out = F.linear(input, self.linear) + z

        flat_shift_dim = np.product(self.shift_dim)
        if out.shape[1] < flat_shift_dim:
            padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[1]], device=out.device)
            out = torch.cat([out, padding], dim=1)
        elif out.shape[1] > flat_shift_dim:
            out = out[:, :flat_shift_dim]

        # handle spatial shifts
        try:
            out = out.view([-1] + self.shift_dim)
        except Exception:
            pass
        return out, target_indices, shifts


def normal_projection_stat(x):
    x = x.view([x.shape[0], -1])
    direction = torch.randn(x.shape[1], requires_grad=False, device=x.device)
    direction = direction / torch.norm(direction)
    projection = torch.matmul(x, direction)

    std, mean = torch.std_mean(projection)
    return std, mean


class ActivationVectorDeformator(nn.Module):
    def __init__(self,
                 instrumented_generator,
                 shift_dims,
                 input_dim,
                 layers: tp.List[str]
                 ):
        super().__init__()
        self.directions_count = input_dim
        self.g = instrumented_generator
        self.layers = [layer_name for layer_name in layers]
        self.g.retain_layers(layers)

        self.vectors = nn.ParameterDict({
            layer_name.replace('.', '/'): nn.Parameter(torch.randn(shape.numel(), input_dim))
            for layer_name, shape in zip(layers, shift_dims)
        })
        self.shapes = {layer_name: shape for layer_name, shape in zip(layers, shift_dims)}
        self.shift_maker = None

    def deform(self, input_):
        b_size = input_.size(0)
        for layer_name in self.layers:
            layer_offset = F.linear(input_, self.vectors[layer_name.replace('.', '/')])
            viewed_offset = layer_offset.view(b_size, *self.shapes[layer_name])
            self.g.edit_layer(layer_name, offset=viewed_offset)

    def forward(self, z):
        target_indices, shifts, basis_shift = self.shift_maker(z)

        self.deform(basis_shift)
        return z, target_indices, shifts


class LA_vector_Deformator(nn.Module):
    def __init__(self):
        super().__init__()
        ...
    def forward(self):
        ...


class WarpedDeformator(nn.Module):
    def __init__(self,
                 shift_dim,
                 num_support_dipoles,
                 support_vectors_dim,
                 learn_alphas=False,
                 learn_gammas=False,
                 gamma=None, 
                 min_shift_magnitude=0.2,
                 max_shift_magnitude=0.5):
        super().__init__()
        self.support_sets = SupportSets(shift_dim,
                                        num_support_dipoles,
                                        support_vectors_dim,
                                        learn_alphas,
                                        learn_gammas,
                                        gamma)
        self.shift_dim = shift_dim
        self.input_dim = shift_dim
        self.min_shift_magnitude = min_shift_magnitude
        self.max_shift_magnitude = max_shift_magnitude
        self.device = 'cuda:0' # TODO: AS ARGUMENT
        self.support_sets.to(self.device)
        self.to(self.device)

    def forward(self, z, basis_shift=None):
        z.to(self.device)
        batch_size = z.size(0)
        
        if basis_shift is None:
            target_indices = torch.randint(
                0, self.shift_dim, [batch_size], device=self.device)


            shift_magnitudes_pos = (self.min_shift_magnitude - self.max_shift_magnitude) * \
                torch.rand(target_indices.size(), device=self.device) + self.max_shift_magnitude
            shift_magnitudes_neg = (self.min_shift_magnitude - self.max_shift_magnitude) * \
                torch.rand(target_indices.size(), device=self.device) - self.min_shift_magnitude
            shift_magnitudes_pool = torch.cat((shift_magnitudes_neg, shift_magnitudes_pos))
            shift_magnitudes_ids = torch.arange(len(shift_magnitudes_pool), dtype=torch.float)

            target_shift_magnitudes = shift_magnitudes_pool[torch.multinomial(input=shift_magnitudes_ids,
                                                                          num_samples=batch_size,
                                                                          replacement=False)]
            target_shift_magnitudes.to(self.device)
            
        else:
            target_indices = torch.argmax(torch.abs(basis_shift), dim=1)
            target_indices.to(self.device)
            target_shift_magnitudes = torch.sum(basis_shift, dim=1, dtype=torch.float)
            target_shift_magnitudes.to(self.device)
        support_sets_mask = torch.zeros([batch_size, self.shift_dim], device=self.device, dtype=torch.float)
        for i, (index, val) in enumerate(zip(target_indices, target_shift_magnitudes)):
            support_sets_mask[i][index] += 1.0
        
        shift = target_shift_magnitudes.reshape(-1, 1) * \
                self.support_sets(support_sets_mask, z)
        z = z + shift
        return z, target_indices, target_shift_magnitudes


class SupportSets(nn.Module):
    """Support Sets class
        TODO: K = as many as the desired interpretable paths
            Each support set contains `num_support_dipoles` support vector dipoles -- i.e., "antipodal" support vectors
            with opposite weights alpha (-1, +1) and the same gamma (scale) parameter. During training the position of
            support vectors are being optimized, while weights alpha and scale parameters gamma are
    """
    def __init__(self, num_support_sets, num_support_dipoles, support_vectors_dim,
                 learn_alphas=False, learn_gammas=False, gamma=None):
        """ SupportSets constructor.
        Args:
            num_support_sets (int)    : number of support sets (each one defining a warping function)
            num_support_dipoles (int) : number of support dipoles per support set (per warping function)
            support_vectors_dim (int) : dimensionality of support vectors (latent space dimensionality, z_dim)
            learn_alphas (bool)       : learn RBF alphas
            learn_gammas (bool)       : learn RBF gammas
            gamma (float)             : RBF gamma parameter (by default set to the inverse of the latent space
                                        dimensionality)
        """
        super(SupportSets, self).__init__()
        self.device = 'cuda:0' # TODO: AS ARGUMENT
        self.to(self.device)
        self.num_support_sets = num_support_sets
        self.num_support_dipoles = num_support_dipoles
        self.support_vectors_dim = support_vectors_dim
        self.learn_alphas = learn_alphas
        self.learn_gammas = learn_gammas
        self.gamma = gamma
        self.loggamma = torch.log(torch.scalar_tensor(self.gamma, device=self.device))

        # TODO: add comment
        self.r = 3.0
        self.r_min = 1.0
        self.r_max = 4.0
        self.r_mean = 0.5 * (self.r_min + self.r_max)
        self.radii = torch.arange(self.r_min, self.r_max, (self.r_max - self.r_min)/self.num_support_sets, device=self.device)
        
        ################################################################################################################
        ##                                                                                                            ##
        ##                                        [ SUPPORT_SETS: (K, N, d) ]                                         ##
        ##                                                                                                            ##
        ################################################################################################################
        # Define learnable parameters ofr RBF support sets:
        #   K sets of N pairs of d-dimensional (antipodal) support vector sets
        self.SUPPORT_SETS = nn.Parameter(data=torch.ones(self.num_support_sets,
                                                         2 * self.num_support_dipoles * self.support_vectors_dim, device=self.device),
                                         requires_grad=True)

        SUPPORT_SETS = torch.zeros(self.num_support_sets, 2 * self.num_support_dipoles, self.support_vectors_dim, device=self.device)
        for k in range(self.num_support_sets):
            SV_set = []
            for i in range(self.num_support_dipoles):
                SV = torch.randn(1, self.support_vectors_dim, device=self.device)
                SV_set.extend([SV, -SV])
            SV_set = torch.cat(SV_set)
            SV_set = self.radii[k] * SV_set / torch.norm(SV_set, dim=1, keepdim=True)
            SUPPORT_SETS[k, :] = SV_set

        # Reshape support sets tensor into a matrix and initialize support sets matrix
        self.SUPPORT_SETS.data = SUPPORT_SETS.reshape(self.num_support_sets,
                                                      2 * self.num_support_dipoles * self.support_vectors_dim).clone()
        # ************************************************************************************************************ #

        ################################################################################################################
        ##                                                                                                            ##
        ##                                            [ ALPHAS: (K, N) ]                                              ##
        ##                                                                                                            ##
        ################################################################################################################
        # REVIEW: Define alphas as parameters (learnable or non-learnable)
        self.ALPHAS = nn.Parameter(data=torch.zeros(self.num_support_sets, 2 * self.num_support_dipoles),
                                   requires_grad=self.learn_alphas)

        for k in range(self.num_support_sets):
            a = []
            for _ in range(self.num_support_dipoles):
                a.extend([1, -1])
            self.ALPHAS.data[k] = torch.Tensor(a)

        ################################################################################################################
        ##                                                                                                            ##
        ##                                            [ GAMMAS: (K, N) ]                                              ##
        ##                                                                                                            ##
        ################################################################################################################
        # Define RBF gammas
        self.LOGGAMMA = nn.Parameter(data=self.loggamma * torch.ones(self.num_support_sets, 1, device=self.device),
                                     requires_grad=self.learn_gammas)

    def forward(self, support_sets_mask, z):
        """TODO: +++
        Args:
            support_sets_mask (torch.Tensor): TODO: +++ -- size: +++
            z (torch.Tensor): input latent codes -- size: TODO: +++
        Returns:
            Normalized grad of f evaluated at given z -- size: (bs, dim).
        """
        # Get RBF support sets batchs
        support_sets_mask.to(self.device)
        z.to(self.device)
        self.SUPPORT_SETS.to(self.device)
        self.ALPHAS.to(self.device)
        
        support_sets_batch = torch.matmul(support_sets_mask, self.SUPPORT_SETS)
        support_sets_batch = support_sets_batch.reshape(-1, 2 * self.num_support_dipoles, self.support_vectors_dim)

        # Get batch of RBF alpha parameters
        alphas_batch = torch.matmul(support_sets_mask, self.ALPHAS).unsqueeze(dim=2)

        # Get batch of RBF gamma/log(gamma) parameters
        if self.learn_gammas:
            gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA).unsqueeze(dim=2))
        else:
            gammas_batch = self.gamma * torch.ones(z.size()[0], 2 * self.num_support_dipoles, 1, device=self.device)

        # Calculate grad of f at z
        D = z.unsqueeze(dim=1).repeat(1, 2 * self.num_support_dipoles, 1) - support_sets_batch
        grad_f = -2 * (alphas_batch * torch.exp(-gammas_batch * (torch.norm(D, dim=2) ** 2).unsqueeze(dim=2)) * D).sum(dim=1)
        grad_f.to(self.device)
        # Return normalized grad of f at z
        return grad_f / torch.norm(grad_f, dim=1, keepdim=True)
