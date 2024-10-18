import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from einops import rearrange
import logging
import os
import time
from torch.utils.data import DataLoader

import tgs
from tgs.models.image_feature import ImageFeature
from tgs.utils.saving import SaverMixin
from tgs.utils.config import parse_structured
from tgs.utils.ops import points_projection
from tgs.utils.misc import load_module_weights
from tgs.utils.typing import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class TGS(torch.nn.Module, SaverMixin):
    @dataclass
    class Config:
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_feature: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        renderer_cls: str = ""  # tgs.models.renderer.GS3DRenderer
        renderer: dict = field(default_factory=dict)

        pointcloud_generator_cls: str = ""
        pointcloud_generator: dict = field(default_factory=dict)

        pointcloud_encoder_cls: str = ""
        pointcloud_encoder: dict = field(default_factory=dict)

    cfg: Config

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)
        torch.cuda.empty_cache()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None

        self.image_tokenizer = tgs.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

        assert self.cfg.camera_embedder_cls == 'tgs.models.networks.MLP'
        weights = self.cfg.camera_embedder.pop("weights") if "weights" in self.cfg.camera_embedder else None
        self.camera_embedder = tgs.find(self.cfg.camera_embedder_cls)(**self.cfg.camera_embedder)
        if weights:
            from tgs.utils.misc import load_module_weights
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        self.image_feature = ImageFeature(self.cfg.image_feature)

        self.tokenizer = tgs.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)

        self.backbone = tgs.find(self.cfg.backbone_cls)(self.cfg.backbone)

        self.post_processor = tgs.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.renderer = tgs.find(self.cfg.renderer_cls)(self.cfg.renderer)
        torch.cuda.empty_cache()

        # pointcloud generator
        self.pointcloud_generator = tgs.find(self.cfg.pointcloud_generator_cls)(self.cfg.pointcloud_generator)

        self.point_encoder = tgs.find(self.cfg.pointcloud_encoder_cls)(self.cfg.pointcloud_encoder)
        torch.cuda.empty_cache()

        # load checkpoint
        if self.cfg.weights is not None:
            self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)
        torch.cuda.empty_cache()

    def _forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # generate point cloud  点云产生
        out = self.pointcloud_generator(batch)
        # print('out1 type:', type(out))
        # print(out)

        pointclouds = out["points"]
        # print('pointclouds type:', type(pointclouds))
        # print(pointclouds)

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]
        torch.cuda.empty_cache()

        # Camera modulation
        # c2w_cond 代表从相机坐标系到世界坐标系的变换矩阵（外参）
        # 分离并调整相机的外部参数(c2w_cond)和内部参数(intrinsic_normed_cond)，
        # 然后将内外参信息合并并通过一个嵌入网络(camera_embedder)转换成特征向量，用于后续对图像特征的条件调制。
        camera_extri = batch["c2w_cond"].view(*batch["c2w_cond"].shape[:-2], -1)
        # intrinsic_normed_cond 则包含了相机的内参
        camera_intri = batch["intrinsic_normed_cond"].view(*batch["intrinsic_normed_cond"].shape[:-2], -1)
        torch.cuda.empty_cache()

        # 相机的内外参信息被合并，并通过一个网络（camera_embedder，一个MLP模块）编码为特征向量
        camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)
        camera_feats = self.camera_embedder(camera_feats)

        # 使用image_tokenizer根据输入的RGB图像和相机特征对图像进行分块和编码，生成图像令牌
        input_image_tokens: Float[Tensor, "B Cit Nit"] = self.image_tokenizer(
            rearrange(batch["rgb_cond"], 'B Nv H W C -> B Nv C H W'),
            modulation_cond=camera_feats,
        )
        input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C', Nv=n_input_views)
        torch.cuda.empty_cache()

        # get image features for projection
        # 利用这些图像令牌和原始图像数据通过image_feature模块进一步提取图像特征，这些特征将用于点云投影。
        image_features = self.image_feature(
            rgb=batch["rgb_cond"],
            mask=batch.get("mask_cond", None),
            feature=input_image_tokens
        )
        torch.cuda.empty_cache()

        # only support number of input view is one：只支持一个输入视图
        c2w_cond = batch["c2w_cond"].squeeze(1)
        intrinsic_cond = batch["intrinsic_cond"].squeeze(1)

        # 通过points_projection函数实现，对生成的点云执行投影操作，将其映射到输入图像的视角下，提取对应的图像特征。
        proj_feats = points_projection(pointclouds, c2w_cond, intrinsic_cond, image_features)
        point_cond_embeddings = self.point_encoder(torch.cat([pointclouds, proj_feats], dim=-1))

        # 使用tokenizer根据点云条件嵌入生成序列令牌，这些令牌代表了场景中结构化的信息。
        # 应用一个序列模型（如Transformer，由self.backbone表示），该模型同时考虑了序列令牌和之前的图像令牌，进行深层次的特征交互和建模
        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size, cond_embeddings=point_cond_embeddings)
        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )
        torch.cuda.empty_cache()

        # 序列模型的输出经过post_processor进一步处理，可能包括解码、上采样等操作，以生成更高维度的场景代码(scene_codes)。
        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))

        # 使用renderer根据场景代码、点云、附加特征以及原始批次中的其他必要信息（如视图方向）来渲染最终的场景输出。
        # 这一步可以生成例如颜色、深度或法线贴图等渲染结果。
        rend_out = self.renderer(scene_codes,
                                 query_points=pointclouds,
                                 additional_features=proj_feats,
                                 **batch)
        torch.cuda.empty_cache()

        return {**out, **rend_out}

    def forward(self, batch):
        # 调用内部前向传播方法：首先通过调用self._forward(batch)执行模型的核心计算逻辑，获取到包含各种输出（如3D网格、渲染好的图像等）的字典out
        out = self._forward(batch)
        # print('out2 type:', type(out))
        # print(out)

        batch_size = batch["index"].shape[0]
        # print('batch_size: ', batch_size)
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                '''
                保存3D网格：对于每个样本（在批量处理的循环中），
                如果该样本的第一个视图索引为0（batch["view_index"][b, 0] == 0），则将该样本对应的3D网格(out["3dgs"][b])
                #保存为.ply文件。.ply是一种常用的3D模型文件格式，可以用来存储点云和三角网格信息。
                保存路径包含了实例ID，确保每个实例的3D网格被唯一标识并保存。
                '''
                out["3dgs"][b].save_ply(self.get_save_path(f"3dgs/{batch['instance_id'][b]}.ply"))
                torch.cuda.empty_cache()

            for index, render_image in enumerate(out["comp_rgb"][b]):
                '''
                保存渲染图像：对于每个样本的每个视图，将渲染出的图像(render_image)保存为PNG格式。
                保存的路径区分了视频序列(video/)和多视图(multi/)两种用途，每个视图的图像根据其视图索引命名。
                这意味着，如果有多个视图，这个过程会为每个视图生成一张或多张渲染图像，
                并保存到相应的目录下。这里使用了save_image_grid方法，
                虽然名字暗示是保存图像网格，但根据提供的信息，实际上似乎是在保存单个图像，
                并且确保了图像的数据格式为"HWC"（高度、宽度、通道数）。
                '''
                view_index = batch["view_index"][b, index]
                self.save_image_grid(
                    f"video/{batch['instance_id'][b]}/{view_index}.png",
                    [
                        {
                            "type": "rgb",
                            "img": render_image,
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                )
                self.save_image_grid(
                    f"multi/{batch['instance_id'][b]}/{view_index}.png",
                    [
                        {
                            "type": "rgb",
                            "img": render_image,
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                )
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    import subprocess
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from tgs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
    parser.add_argument("--config", default="tgs.yaml", help="path to config file")
    # parser.add_argument("--out", default="leaves_data_output", help="path to output folder")
    # parser.add_argument("--out", default="leaves_data_output2", help="path to output folder")
    parser.add_argument("--out", default="./叶片/zsy", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.5, type=float, help="distance between camera center and scene center")

    args, extras = parser.parse_known_args()

    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)

    model_path = "checkpoints/tgs.ckpt"
    cfg.system.weights = model_path
    model = TGS(cfg=cfg.system).to(device)
    model.set_save_dir(args.out)
    torch.cuda.empty_cache()
    print("load model ckpt done.")

    # run image segmentation for images
    # 以递归查找目录下的所有图像文件
    image_paths = []
    for root, dirs, files in os.walk(cfg.data.image_list[0]):  # 假设image_list中只有一个目录路径
        for file in files:
            if file.lower().endswith(('.JPG', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    # print(image_paths)
    segmented_image_list = []
    timer.start("Processing images")
    for image_path in image_paths:
        # print(image_path)
        filepath, ext = os.path.splitext(image_path)
        save_path = os.path.join(filepath + "_rgba.png")
        segmented_image_list.append(save_path)
        # print(save_path)
        timer.start("Processing images")
        cmd = [
            "python",  # 或者使用 "python3" 或完整的解释器路径
            "image_preprocess/run_sam.py",
            "--image_path", image_path,
            "--save_path", save_path,
        ]
        subprocess.run(cmd, check=True)
        timer.end("Processing images")

        cfg.data.image_list = segmented_image_list
        # torch.cuda.empty_cache()
    timer.end("Processing images")
    print("segment image done")
    # torch.cuda.empty_cache()

    # 设定相机到场景中心的距离
    cfg.data.cond_camera_distance = args.cam_dist  # default=1.1
    cfg.data.eval_camera_distance = args.cam_dist
    # 数据加载时指定相机参数
    dataset = CustomImageOrbitDataset(cfg.data)
    # torch.cuda.empty_cache()
    print("dataset done")
    dataloader = DataLoader(dataset,
                            batch_size=cfg.data.eval_batch_size,
                            num_workers=8,
                            shuffle=False,
                            collate_fn=dataset.collate
                            )
    # torch.cuda.empty_cache()
    print("dataloader done")

    print("Starting image to 3D processing...")
    timer.start("Running model")
    for idx, batch in enumerate(tqdm(dataloader, desc="Images", total=len(dataloader))):
        print(f"Processing image {idx + 1}/{len(dataloader)}")  # 打印当前处理的图像编号和总数
        batch = todevice(batch)
        model(batch)
        # torch.cuda.empty_cache()
    timer.end("Running model")

    timer.start("Exporting video")
    model.save_img_sequences(
        "video",
        "(\d+)\.png",
        save_format="mp4",
        fps=30,
        delete=True,
    )
    timer.end("Exporting video")
    torch.cuda.empty_cache()
    print("All images processed and video saved.")
