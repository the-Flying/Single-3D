import os
from PIL import Image
from tqdm import tqdm


def convert_images_to_rgb(input_folder, output_folder=None):
    # 确保输入文件夹存在
    assert os.path.exists(input_folder), f"Folder '{input_folder}' does not exist."

    # 如果指定了输出文件夹，则创建它
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    # 获取所有图像文件的列表
    image_files = [filename for filename in os.listdir(input_folder) if
                   os.path.isfile(os.path.join(input_folder, filename)) and filename.lower().endswith(
                       ('.png', '.jpg', '.jpeg'))]

    # 创建一个 tqdm 进度条对象
    progress_bar = tqdm(image_files, desc="Converting Images to RGB", unit="image")

    # 计数器用于新的文件名
    counter = 0

    # 遍历文件夹中的所有文件
    for filename in progress_bar:
        # 构建完整的文件路径
        file_path = os.path.join(input_folder, filename)

        # 打开图像
        with Image.open(file_path) as img:
            # 如果图像不是 RGB 模式，则转换为 RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 新的文件名
            new_filename = f"{counter}.jpg"  # 以 .jpg 结尾
            if output_folder is not None:
                new_file_path = os.path.join(output_folder, new_filename)
            else:
                new_file_path = os.path.join(input_folder, new_filename)

            # 保存图像，保持原文件名不变
            img.save(new_file_path)

        # 更新进度条
        progress_bar.update(1)

        # 增加计数器
        counter += 1

    # 关闭进度条
    progress_bar.close()


def main():
    input_folder = "C:/Users/lab/Desktop/CodeReconstruction/TriplaneGaussian-code/diffusion_enhance/Potato___healthy"
    output_folder = "C:/Users/lab/Desktop/CodeReconstruction/TriplaneGaussian-code/diffusion_enhance/Potato_Healthy"
    convert_images_to_rgb(input_folder, output_folder)
    print("All images in the folder have been converted to RGB mode and renamed.")


if __name__ == "__main__":
    main()
