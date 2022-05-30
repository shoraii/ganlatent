import numpy as np
import torch
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image
import io
import os

from torch_tools.visualization import to_image

from utils import make_noise, one_hot, is_conditional


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


@torch.no_grad()
def interpolate(
    generator, deformator, z, shifts_r, shifts_count, dim, with_central_border=False
):
    shifted_images = []
    
    device = next(generator.parameters()).device
    
    if is_conditional(generator):
        if hasattr(generator, 'model'):
            classes = torch.from_numpy(np.random.choice(generator.model.target_classes.cpu(), z.size(0))).to(device)
        else:
            classes = torch.from_numpy(np.random.choice(generator.target_classes.cpu(), z.size(0))).to(device)
        
        
    for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
        basis_shift = one_hot(deformator.input_dim, shift, dim).cuda().unsqueeze(0).repeat(z.size(0), 1)
        shifted_z, *_ = deformator(z, basis_shift=basis_shift)

        if is_conditional(generator):
            shifted_image = generator(shifted_z, classes).detach().cpu()[0]
        else:
            shifted_image = generator(shifted_z).detach().cpu()[0]

        shifted_image = (shifted_image + 1) / 2

        if shift == 0.0 and with_central_border:
            shifted_image = add_border(shifted_image)

        shifted_images.append(shifted_image)

    return shifted_images


def add_border(tensor):
    border = 3
    for ch in range(tensor.shape[0]):
        color = 1.0 if ch == 0 else -1
        tensor[ch, :border, :] = color
        tensor[ch, -border:, ] = color
        tensor[ch, :, :border] = color
        tensor[ch, :, -border:] = color
    return tensor


@torch.no_grad()
def make_interpolation_chart(generator, deformator=None, z=None,
                             shifts_r=10.0, shifts_count=5,
                             directions=None, dims_count=10, texts=None, **kwargs):
    with_deformation = deformator is not None

    if with_deformation:
        deformator_is_training = deformator.training
        deformator.eval()

    z = z if z is not None else make_noise(1, generator.dim_z).cuda()


    imgs = []
    if directions is None:
        directions = range(dims_count)
    for current_direction in directions:
        imgs.append(interpolate(generator, deformator, z, shifts_r, shifts_count, current_direction))

    rows_count = len(imgs)
    fig, axs = plt.subplots(rows_count, **kwargs)

    if texts is None:
        texts = directions
    for ax, shifts_imgs, text in zip(axs, imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
        ax.text(-20, 21, str(text), fontsize=10)

    if with_deformation and deformator_is_training:
        deformator.train()

    return fig


@torch.no_grad()
def inspect_all_directions(G, deformator, out_dir, zs=None, num_z=3, shifts_r=8.0):
    os.makedirs(out_dir, exist_ok=True)

    step = 20
    max_dim = deformator.input_dim
    
    try:
        dim_z = G.dim_z
    except Exception as e:
        dim_z = G.model.dim_z
    
    zs = zs if zs is not None else make_noise(num_z, dim_z).cuda()
    shifts_count = zs.shape[0]

    for start in range(0, max_dim - 1, step):
        imgs = []
        dims = range(start, min(start + step, max_dim))
        for z in zs:
            z = z.unsqueeze(0)
            fig = make_interpolation_chart(
                G, deformator=deformator, z=z,
                shifts_count=shifts_count, directions=dims, shifts_r=shifts_r,
                dpi=250, figsize=(int(shifts_count * 4.0), int(0.5 * step) + 2))
            fig.canvas.draw()
            plt.close(fig)
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # crop borders
            nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
            img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
            imgs.append(img)

        out_file = os.path.join(out_dir, '{}_{}.jpg'.format(dims[0], dims[-1]))
        print('saving chart to {}'.format(out_file))
        Image.fromarray(np.hstack(imgs)).save(out_file)


def gen_animation(G, deformator, direction_index, out_file, z=None, size=None, r=8):
    import imageio

    if z is None:
        z = torch.randn([1, G.dim_z], device='cuda')
    interpolation_deformed = interpolate(
        G, z, shifts_r=r, shifts_count=5,
        dim=direction_index, deformator=deformator, with_central_border=False)

    resize = Resize(size) if size is not None else lambda x: x
    img = [resize(to_image(torch.clamp(im, -1, 1))) for im in interpolation_deformed]
    imageio.mimsave(out_file, img + img[::-1])
