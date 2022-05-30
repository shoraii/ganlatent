import torch
import torch.nn.functional as F
from collections import defaultdict

from utils_common.class_registry import ClassRegistry

loss_registry = ClassRegistry()


@loss_registry.add_to_registry("ce_loss")
def cross_entopy_one_direction(logits_directions, ground_truth_directions):
    loss = F.cross_entropy(logits_directions, ground_truth_directions, reduction='none')
    return loss


@loss_registry.add_to_registry("mae_magnitude_scalar")
def mae_magnitude_scalar(pred, gt):
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)

    if gt.ndim == 1:
        gt = gt.unsqueeze(1)

    loss = torch.abs(pred - gt)

    if loss.ndim == 2:
        loss = loss.squeeze(1)

    return loss


@loss_registry.add_to_registry("mse_magnitude_scalar")
def mse_magnitude_scalar(pred, gt):
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)

    if gt.ndim == 1:
        gt = gt.unsqueeze(1)

    loss = F.mse_loss(pred, gt, reduction='none')

    if loss.ndim == 2:
        loss = loss.squeeze(1)

    return loss


@loss_registry.add_to_registry("mse_shift_reg")
def mse_magnitude_scalar(z, z_edited):
    if z.ndim == 1:
        z = z.unsqueeze(1)

    if z_edited.ndim == 1:
        z_edited = z_edited.unsqueeze(1)

    loss = F.mse_loss(z, z_edited)

    if loss.ndim == 2:
        loss = loss.squeeze(1)

    return loss


@loss_registry.add_to_registry("dyn_mse_shift_reg")
def dyn_mse_magnitude_scalar(z, z_edited, shifts):
    if z.ndim == 1:
        z = z.unsqueeze(1)

    if z_edited.ndim == 1:
        z_edited = z_edited.unsqueeze(1)

    loss = F.mse_loss(z, z_edited, reduction='none')
    loss = torch.mean(loss, dim=1)
    loss = loss / (shifts * shifts)

    return torch.mean(loss)


@loss_registry.add_to_registry("gram_orthogonal")
def gram_orthogonal_loss(generator, layers, target_indices):
    for layer, retained_features in generator.retained_features().items():
        pass


class GramOrthogonalLoss:
    def __init__(self, generator):
        self.g = generator

    def __call__(self, target_indices):
        feature_store = defaultdict(list)

        for key, features in self.g.retained_features().items():
            for idx, ind in enumerate(target_indices):
                feature_store[ind.item()].append(features[idx])

        dir_num = 128
        pre_gram = torch.zeros(dir_num, 1536 * 8 * 8)

        for key, tensors in feature_store.items():
            pre_gram[key] = sum(tensors).view(-1)

        gram_ = pre_gram @ pre_gram.t()

        return (gram_ - torch.diag(torch.diag(gram_))).abs().sum() / 2.


class MultiLossContainer:
    def __init__(self, loss_funcs, loss_coefs):
        self.loss_funcs = loss_funcs
        num_losses = len(self.loss_funcs)
        self.loss_coefs = loss_coefs
        if not loss_coefs:
            self.loss_coefs = [1] * num_losses
        assert len(self.loss_coefs) == num_losses


class LatentDirectionSearchLoss(MultiLossContainer):
    shift_losses = (
        'mae_magnitude_scalar',
        'mse_magnitued_scalar',
        'mae_magnitude_vector',
        'mse_magnitude_vector'
    )

    ind_losses = ('ce_loss', )
    gram_losses = ('gram_orthogonal', )
    
    shift_reg = ('mse_shift_reg', )
    
    dyn_shift_reg = ('dyn_mse_shift_reg', )

    def __call__(
        self, logits, shift_prediction, target_indices, target_shifts, z, z_edited,
            instrumented_generator
    ):
        losses = defaultdict(float)
        batch_size = logits.size(0)
        for func, coef in zip(self.loss_funcs, self.loss_coefs):
            if func in self.shift_losses:
                vector_loss = loss_registry[func](shift_prediction, target_shifts)

                for idx, (trg, loss) in enumerate(zip(target_indices, vector_loss)):
                    key_loss = func + "/" + str(trg.item())
                    losses[key_loss] = vector_loss[idx]
                    losses[func] += losses[key_loss] / batch_size

                losses["total"] += coef * losses[func]

            elif func in self.ind_losses:
                ind_loss = loss_registry[func](logits, target_indices)

                for idx, (trg, loss) in enumerate(zip(target_indices, ind_loss)):
                    key_loss = func + "/" + str(trg.item())
                    losses[key_loss] = ind_loss[idx]
                    losses[func] += losses[key_loss] / batch_size

                losses["total"] += coef * losses[func]
            elif func in self.shift_reg:
                reg_loss = loss_registry[func](z, z_edited)
                losses[func] = reg_loss / batch_size
                losses["total"] += coef * losses[func]
            elif func in self.dyn_shift_reg:
                reg_loss = loss_registry[func](z, z_edited, target_shifts)
                losses[func] = reg_loss / batch_size
                losses["total"] += coef * losses[func]
            else:
                raise NotImplementedError(f"{func} loss is not implemented!")

#         for func in self.loss_funcs + ["total"]:
#             losses[f"agg/{func}"] = losses[func]

        return losses
