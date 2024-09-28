from pathlib import Path

from PIL import Image

from .backend_diffusers import txt2img


def 生(model, VAE, sampler_name, scheduler):
    Path('测试样例').mkdir(exist_ok=True)
    测试用prompt = [
        '1girl, blonde hair, twintails, blue dress, white apron, smile, closed mouth, tachi-e, fullbody, white background, Alice in glitterworld',
        '1girl, black twintails, school uniform, outdoors, street, fullbody, black pantyhose, holding phone, looking at phone',
        '1girl, twintails, cat ears, maid, maid headdress, holding tray, white pantyhose, indoors, kitchen',
        '1girl, twintails, samurai, japanese armor, seiza, indoors, holding cup of milk',
    ]
    images = []
    for i, prompt in enumerate(测试用prompt):
        images.append(txt2img({
            'prompt': prompt,
            'negative_prompt': 'worst quality, low quality',
            'seed': 1 + i,
            'width': 640,
            'height': 1280,
            'steps': 30,
            'sampler_name': sampler_name,
            'scheduler': scheduler,
            'cfg_scale': 7,
            'override_settings': {
                'sd_model_checkpoint': model,
                'sd_vae': VAE,
            },
            'model_type': 'sdxl',
            'batch_size': 1,
            'n_iter': 1,
        }, 缓存=False, return_pil=True)[0])
    new_image = Image.new('RGB', (640 * len(images), 1280))
    for i, image in enumerate(images):
        new_image.paste(image, (i * 640, 0))
    new_image.save(f'测试样例/{model}.png')


if __name__ == '__main__':
    生('ConfusionXL_R32', 'sdxl_vae_0.9.safetensors', 'DPM++ 2M', 'Karras')
