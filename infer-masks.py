import logging
import os
import time

import torch
import argparse
import subprocess
from tgs.utils.misc import get_device

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default="1")  # 修改默认输入路径
    parser.add_argument("--output_folder", default="1")  # 新增输出路径参数
    args = parser.parse_args()

    device = get_device()

    # Ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Process each image in the input folder
    timer.start("Processing images")
    counter = 1
    for file_name in os.listdir(args.input_folder):
        if file_name.endswith(('.png', '.PNG', '.jpg', '.JPG', '.jpeg')):
            input_path = os.path.join(args.input_folder, file_name)
            output_path = os.path.join(args.output_folder, f"{counter}_processed.png")  # 使用新的输出路径)

            # print(image_path)
            filepath, ext = os.path.splitext(input_path)
            save_path = os.path.join(filepath + "_masks.png")
            timer.start("Processing images")
            cmd = [
                "python",  # 或者使用 "python3" 或完整的解释器路径
                "image_preprocess/run_sam.py",
                "--image_path", input_path,
                "--save_path", save_path,
            ]
            subprocess.run(cmd, check=True)
        timer.end("Processing images")
        torch.cuda.empty_cache()

    timer.end("Processing images")
    print("segment image done")
    torch.cuda.empty_cache()
