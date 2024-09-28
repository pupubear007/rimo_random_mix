import random
from pathlib import Path

from tqdm import tqdm
from PIL import Image

from .common import ml_danbooru标签2, 要测的标签
from .backend_diffusers import txt2img
from .生成测试样例 import 生


存图文件夹 = Path('out_多标签')
存图文件夹.mkdir(exist_ok=True)


def 评测模型(model, VAE, m, n_iter, use_tqdm=True, extra_prompt='', tags_seed=0, 生成测试样例=False):
    rd = random.Random(tags_seed)
    本地记录 = []
    iterator = range(n_iter)
    sampler = 'DPM++ 2M_Karras'
    sampler_name, scheduler = sampler.split('_')
    if 生成测试样例:
        生(model, VAE, sampler_name, scheduler)
    if use_tqdm:
        iterator = tqdm(iterator, ncols=70, desc=f'{m}-{model[:10]}')
    for index in iterator:
        negative_prompt = random.choice(['worst quality, low quality', 'worst quality, low quality, blurry, greyscale, monochrome'])
        steps = 18+random.randint(0, 12)
        cfg_scale = random.choice([6, 7, 8])
        width = 512+random.randint(1, 5)*64
        height = 512+random.randint(1, 5)*64
        标签组 = rd.sample(要测的标签, m)
        标签组 = [i.strip().replace(' ', '_') for i in 标签组]
        参数 = {
            'prompt': f'1 girl, {", ".join(标签组)}'+extra_prompt,
            'negative_prompt': negative_prompt,
            'seed': random.randint(0, 2**16),
            'width': width,
            'height': height,
            'steps': steps,
            'sampler_name': sampler_name,
            'scheduler': scheduler,
            'cfg_scale': cfg_scale,
            'override_settings': {
                'sd_model_checkpoint': model,
                'sd_vae': VAE,
            },
            'model_type': 'sdxl',
        }
        数量参数 = {
            'batch_size': 4,
            'n_iter': 1,
        }
        图s = txt2img(数量参数 | 参数, 缓存=False, return_pil=True)
        n = len(图s)
        预测标签 = ml_danbooru标签2(图s)
        录 = {
            '分数': [[i.get(j, 0) for j in 标签组] for i in 预测标签],
            '总数': n,
            '标签组': 标签组,
            '参数': 参数,
            '预测标签': {str(k): v for k, v in enumerate(预测标签)},
        }
        本地记录.append(录)
    return 本地记录


if __name__ == '__main__':
    import numpy as np
    for 文件名 in ['ConfusionXL_R16', 'ConfusionXL_R8']:
        结果 = 评测模型(文件名, 'sdxl_vae_0.9.safetensors', 32, n_iter=100, use_tqdm=True, tags_seed=114514)
        m = []
        tm = []
        for dd in 结果:
            m.extend(dd['分数'])
            for k, v in dd['预测标签'].items():
                tm += v
        mm = np.array(m)
        acc = (mm > 0.001).sum() / len(mm.flatten())
        print(文件名, acc)
