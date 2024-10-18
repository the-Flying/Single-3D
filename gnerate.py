from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
import os

model_id = "ddpm-model-256"

# 生成的图像放的位置
img_path = 'results_rice' + '/' + model_id + '-img'

if not os.path.exists(img_path):
    os.mkdir(img_path)

device = "cuda"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(
    model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
ddpm.to(device)
for i in range(1000):
    # run pipeline in inference (sample random noise and denoise)
    image = ddpm().images[0]
    # save image
    # 不修改格式
    image.save(os.path.join(img_path, f'{i}.png'))
    # 改成单通道
    # image.convert('L').save(os.path.join(img_path, f'{i}.png'))
    # 看看跑到哪里了
    if i % 10 == 0:
        print(f"i={i}")
