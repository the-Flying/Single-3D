import math
import os
from functools import partial  # 偏函数 https://www.runoob.com/w3cnote/python-partial.html
from inspect import \
    isfunction  # inspect模块https://www.cnblogs.com/yaohong/p/8874154.html主要提供了四种用处：1.对是否是模块、框架、函数进行类型检查 2.获取源码 3.获取类或者函数的参数信息 4.解析堆栈
from pathlib import Path
from PIL import Image

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
# 展示从噪声生成图像的过程
import torch
import torch.nn.functional as F
from einops import rearrange  # einops把张量的维度操作具象化，让开发者“想出即写出
from torch import nn, einsum  # einsum很方便的实现复杂的张量操作 https://zhuanlan.zhihu.com/p/361209187
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, RandomHorizontalFlip, ToPILImage
from torchvision.transforms.functional import to_tensor, resize, center_crop
from torchvision.utils import save_image
from tqdm.auto import tqdm  # 进度条


# x是否为None，不是None则返回True，是None则返回False
def exists(x):
    return x is not None


# 如果val非None则返回val，否则(如果d为函数则返回d(),否则返回d)
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# 残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# 上采样
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


# 下采样
def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


# 一种位置编码，前一半sin后一半cos
# eg：维数dim=5，time取1和2两个时间
# layer = SinusoidalPositionEmbeddings(5)
# embeddings = layer(torch.tensor([1,2]))
# return embeddings的形状是(2,5),第一行是t=1时的位置编码，第二行是t=2时的位置编码
# 额外连接(transformer原作位置编码实现)：https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Block类，先卷积后GN归一化后siLU激活函数，若存在scale_shift则进行一定变换
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)  # GN归一化 https://zhuanlan.zhihu.com/p/177853578
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


# 例：dim=8，dim_out=16,time_emb_dim=2, groups=8
# Block = ResnetBlock(8, 16, time_emb_dim=2, groups=8)
# a = torch.ones(1, 8, 64, 64)
# b = torch.ones(1, 2)
# result = Block(a, b)

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # 如果time_emb_dim存在则有mlp层
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out,
                                  1) if dim != dim_out else nn.Identity()  # nn.Identity()有 https://blog.csdn.net/artistkeepmonkey/article/details/115067356

    def forward(self, x, time_emb=None):
        h = self.block1(x)  # torch.Size([1, 16, 64, 64])

        if exists(self.mlp) and exists(time_emb):
            # time_emb为torch.Size([1, 2])
            time_emb = self.mlp(time_emb)  # torch.Size([1, 16])
            # rearrange(time_emb, "b c -> b c 1 1")为torch.Size([1, 16, 1, 1])
            h = rearrange(time_emb, "b c -> b c 1 1") + h  # torch.Size([1, 16, 64, 64])

        h = self.block2(h)  # torch.Size([1, 16, 64, 64])
        return h + self.res_conv(x)  # return最后补了残差连接 # torch.Size([1, 16, 64, 64])


# 可以参考class ResnetBlock进行理解
class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        # 如果time_emb_dim存在则有mlp层
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)  # qkv为一个元组，其中每一个元素的大小为torch.Size([b, hidden_dim, h, w])
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )  # qkv中每个元素从torch.Size([b, hidden_dim, h, w])变为torch.Size([b, heads, dim_head, h*w])
        q = q * self.scale  # q扩大dim_head**-0.5倍

        sim = einsum("b h d i, b h d j -> b h i j", q, k)  # sim有torch.Size([b, heads, h*w, h*w])
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)  # attn有torch.Size([b, heads, h*w, h*w])

        out = einsum("b h i j, b h d j -> b h i d", attn,
                     v)  # [b, heads, h*w, h*w]和[b, heads, dim_head, h*w] 得 out为[b, heads, h*w, dim_head]
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)  # 得out为[b, hidden_dim, h, w]
        return self.to_out(out)  # 得 [b, dim, h, w]


# 和class Attention几乎一致
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


# 先norm后fn
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
            self,
            dim,  # 下例中，dim=image_size=28
            init_dim=None,  # 默认为None，最终取dim // 3 * 2
            out_dim=None,  # 默认为None，最终取channels
            dim_mults=(1, 2, 4, 8),
            channels=3,  # 通道数默认为3
            with_time_emb=True,  # 是否使用embeddings
            resnet_block_groups=8,  # 如果使用ResnetBlock，groups=resnet_block_groups
            use_convnext=True,  # 是True使用ConvNextBlock，是Flase使用ResnetBlock
            convnext_mult=2,  # 如果使用ConvNextBlock，mult=convnext_mult
    ):
        super().__init__()
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]  # 从头到尾dim组成的列表
        in_out = list(zip(dims[:-1], dims[1:]))  # dim对组成的列表
        # 使用ConvNextBlock或ResnetBlock
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])  # 初始化下采样网络列表
        self.ups = nn.ModuleList([])  # 初始化上采样网络列表
        num_resolutions = len(in_out)  # dim对组成的列表的长度

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)  # 是否到了最后一对

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# sqrt_alphas_cumprod = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# x_start = torch.ones([1, 3, 8, 8])
# out = extract(a=sqrt_alphas_cumprod, t=torch.tensor([5]), x_shape=x_start.shape)
# print(out.shape)
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t):
    # add noise
    x_noisy = q_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image


# use seed for reproducability
torch.manual_seed(0)  # torch.manual_seed(0)


def save_image(img, path):
    # img = reverse_transform(img)
    img.save(path)


# pytorch官方的一个画图函数
# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs, save_dir, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)

    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            save_path = os.path.join(save_dir, f'image_{row_idx}_{col_idx}.png')
            save_image(img, save_path)

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


# x = np.linspace(1, 1001, 1000)
# timesteps = 1000
#
# fig, ax = plt.subplots()  # 创建图实例
# ax.plot(x, (cosine_beta_schedule(timesteps, s=0.008) / 50).numpy(), label='cosine')
# ax.plot(x, linear_beta_schedule(timesteps).numpy(), label='linear')
# ax.plot(x, quadratic_beta_schedule(timesteps).numpy(), label='quadratic')
# ax.plot(x, sigmoid_beta_schedule(timesteps).numpy(), label='sigmoid')
# plt.legend()
# plt.show()

timesteps = 200

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

dir_path = './leaves_data/eb.jpg'
image_cv = cv2.imread(dir_path)
# 将BGR图像转换为RGB
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
# # 将NumPy数组转换为PIL Image对象
# image = Image.fromarray(image_rgb)
# 直接转换为PyTorch张量
image = to_tensor(image_rgb)
# # 使用matplotlib显示图像
# plt.imshow(image)
# plt.show()

image_size = 256
transform = Compose([
    Lambda(lambda img: resize(img, [image_size, image_size])),  # 变为形状为256*256
    Lambda(lambda img: center_crop(img, [image_size, image_size])),  # 中心裁剪
    Lambda(lambda t: (t * 2) - 1),  # 变为[-1,1]范围

])
x_start = transform(image).unsqueeze(0)
# print(x_start.shape)

reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),
    Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
    Lambda(lambda t: t * 255.),
    Lambda(lambda t: t.numpy().astype(np.uint8)),
    ToPILImage(),
])

# # 处理后的图片
# plt.imshow(reverse_transform(x_start.squeeze()))
# plt.show()

# # 观察结果多次前向传播后的图像
save_dir = './noisy_images'
os.makedirs(save_dir, exist_ok=True)
plot([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]], save_dir)


# #################################################################################################
# # 去噪过程
# #################################################################################################
# # 数据集类
# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_path).convert("RGB")  # 假设图像为彩色，转换为 RGB
#         if self.transform:
#             image = self.transform(image)
#         return image
#
#
# # 定义损失函数 p_losses：三种损失函数有l1，l2和huber，默认为l1
# def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
#     if noise is None:
#         noise = torch.randn_like(x_start)
#
#     x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
#     predicted_noise = denoise_model(x_noisy, t)
#
#     if loss_type == 'l1':
#         loss = F.l1_loss(noise, predicted_noise)
#     elif loss_type == 'l2':
#         loss = F.mse_loss(noise, predicted_noise)
#     elif loss_type == "huber":
#         loss = F.smooth_l1_loss(noise, predicted_noise)
#     else:
#         raise NotImplementedError()
#     return loss
#
#
# # load dataset from the hub：加载了 Fashion MNIST 数据集
# # 论文：Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms，
# # 其中值得我们关注的是数据制作的方法：最原始图片是背景为浅灰色的，
# # 分辨率为7621000 的JPEG图片。然后经过resampled 到 5173 的彩色图片。
# from datasets import load_dataset
# dataset = load_dataset("fashion_mnist")
# image_size2 = 256
# channels = 1
# batch_size = 128
#
# # define image transformations (e.g. using torchvision)
# transform = Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda t: (t * 2) - 1)
# ])
#
#
# # define function
# def transforms(examples):
#     examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
#     del examples["image"]
#
#     return examples
#
#
# # # 得到变换之后的数据集
# transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
# # create dataloader
# dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
# print("data loader done!!!")
#
#
# @torch.no_grad()
# def p_sample(model, x, t, t_index):
#     # p_sample(model, img, torch.full((b,),i,device=device,dtype=torch.long),i)
#     betas_t = extract(betas, t, x.shape)
#     sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
#     sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
#
#     # Equation 11 in the paper
#     # Use our model (noise predictor) to predict the mean
#     model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
#
#     if t_index == 0:
#         return model_mean
#     else:  # 加一定的噪声
#         posterior_variance_t = extract(posterior_variance, t, x.shape)
#         noise = torch.randn_like(x)
#         # Algorithm 2 line 4:
#         return model_mean + torch.sqrt(posterior_variance_t) * noise
#
#
# @torch.no_grad()
# def p_sample_loop(model, shape):
#     # 从噪声中逐步采样
#     device = next(model.parameters()).device
#
#     b = shape[0]
#
#     img = torch.randn(shape, device=device)
#     imgs = []
#
#     for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
#         img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
#         imgs.append(img.cpu().numpy())
#     return imgs
#
#
# # @torch.no_grad()
# def sample(model, image_size, batch_size=16, channels=3):
#     return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
#
#
# # 例如num = 10， divisor = 3，得[3,3,3,1]
# def num_to_groups(num, divisor):
#     groups = num // divisor
#     remainder = num % divisor
#     arr = [divisor] * groups
#     if remainder > 0:
#         arr.append(remainder)
#     return arr
#
#
# # 设置了训练相关的参数，包括结果存储路径、使用的设备（CPU 或 GPU）、模型、优化器等。
# results_folder = Path("./noisy_results")
# results_folder.mkdir(exist_ok=True)  # https://zhuanlan.zhihu.com/p/317254621
# save_and_sample_every = 1000
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = Unet(
#     dim=image_size2,
#     channels=channels,
#     dim_mults=(1, 2, 4)
# )
# model.to(device)
# optimizer = Adam(model.parameters(), lr=1e-3)
# epochs = 5
#
# # 实现了训练循环，包括前向传播、计算损失、反向传播和参数更新等
# for epoch in range(epochs):
#     for step, batch in enumerate(dataloader):
#         optimizer.zero_grad()  # 优化器数值清零
#
#         batch_size = batch["pixel_values"].shape[0]
#         batch = batch["pixel_values"].to(device)
#
#         # Algorithm 1 line 3: sample t uniformally for every example in the batch
#         t = torch.randint(0, timesteps, (batch_size,), device=device).long()  # 随机取t
#
#         loss = p_losses(model, batch, t, loss_type="huber")
#
#         if step % 100 == 0:
#             print("Loss:", loss.item())
#
#         loss.backward()
#         optimizer.step()
#         # save generated images：保存和采样生成的图像
#         if step != 0 and step % save_and_sample_every == 0:
#             milestone = step // save_and_sample_every
#             batches = num_to_groups(4, batch_size)
#             all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
#             all_images = torch.cat(all_images_list, dim=0)
#             all_images = (all_images + 1) * 0.5
#             save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)
#
# print("done!")
#
# # 展示了从噪声到最终图像的生成过程。
# # sample 64 images
# samples = sample(model, image_size=image_size2, batch_size=64, channels=channels)
#
# # show a random one
# random_index = 5
# plt.imshow(samples[-1][random_index].reshape(image_size2, image_size2, channels))
#
# random_index = 53
#
# fig = plt.figure()
# ims = []
# for i in range(timesteps):
#     im = plt.imshow(samples[i][random_index].reshape(image_size2, image_size2, channels), animated=True)
#     ims.append([im])
#
# animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# # animate.save('diffusion.gif')
# plt.show()
