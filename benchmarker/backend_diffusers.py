import io
import time
from pathlib import Path

import torch
from transformers import T5EncoderModel
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline, StableDiffusionKDiffusionPipeline, FluxTransformer2DModel, FluxPipeline, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from compel import Compel, ReturnedEmbeddingsType
from optimum.quanto import freeze, qfloat8, quantize

import rimo_storage.cache


from DeepCache import DeepCacheSDHelper


model_dir = 'R:/stable-diffusion-webui-master/models'



_l = torch.load
def _假load(*t, **d):
    d.pop('weights_only', None)
    return _l(*t, **d)
torch.load = _假load


_is_skip_step = DeepCacheSDHelper.is_skip_step
def is_skip_step(self, block_i, layer_i, blocktype = "down"):
    self.start_timestep = self.cur_timestep if self.start_timestep is None else self.start_timestep # For some pipeline that the first timestep != 0
    if self.cur_timestep-self.start_timestep < self.start_step:
        return False
    return _is_skip_step(self, block_i, layer_i, blocktype)
DeepCacheSDHelper.is_skip_step = is_skip_step


class 超StableDiffusionKDiffusionPipeline:
    def __init__(self, path, vae_path=None):
        p = {}
        if vae_path:
            p['vae'] = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16, weights_only=False).to("cuda")
        pipe0 = StableDiffusionPipeline.from_single_file(path, **p, torch_dtype=torch.float16)
        c = pipe0.components
        c.pop('image_encoder')
        self._pipe = StableDiffusionKDiffusionPipeline(**c).to('cuda')
        self._pipe.set_scheduler('sample_dpmpp_2m')
        self._compel = Compel(tokenizer=self._pipe.tokenizer, text_encoder=self._pipe.text_encoder, truncate_long_prompts=False)

    def __call__(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        **d,
    ):
        conditioning = self._compel(prompt)
        negative_conditioning = self._compel(negative_prompt)
        [conditioning, negative_conditioning] = self._compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
        return self._pipe.__call__(
            prompt_embeds=conditioning,
            negative_prompt_embeds=negative_conditioning,
            **d,
        )


class 超StableDiffusionXLPipeline:
    def __init__(self, path, vae_path=None, 串行化vae=True, 使用deepcache=True, 保存中间结果=False):
        p = {}
        if vae_path:
            p['vae'] = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16, scaling_factor=0.13025).to("cuda")
        dpmpp_2m = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')
        self._pipe = StableDiffusionXLPipeline.from_single_file(
            path,
            torch_dtype=torch.float16,
            scheduler=dpmpp_2m,
            **p,
        ).to("cuda")
        self._pipe.set_progress_bar_config(disable=True)
        if 串行化vae:
            self._pipe.enable_vae_slicing()
        if 使用deepcache:
            helper = DeepCacheSDHelper(pipe=self._pipe)
            helper.start_step = 6
            helper.set_params(cache_interval=2)
            helper.enable()
        if 保存中间结果:
            _s = self._pipe.scheduler.step
            def 假step(noise_pred, t, latents, **d):
                m = self._pipe.scheduler.timesteps.max()
                denoised_latents = (latents-noise_pred * (t / m))[0]    # 这个实现有点问题，不过还算能看，就先这样吧
                denoised_latents = denoised_latents.reshape([1, *denoised_latents.shape])
                image = self._pipe.image_processor.postprocess(
                    self._pipe.vae.to(torch.float32).decode(denoised_latents.to(torch.float32) / self._pipe.vae.config.scaling_factor, return_dict=False)[0],
                    output_type='pil',
                )[0]
                self.step_image.append(image)
                self._pipe.vae.to(torch.float16)
                return _s(noise_pred, t, latents, **d)
            self._pipe.scheduler.step = 假step
        self.compel = Compel(truncate_long_prompts=False, tokenizer=[self._pipe.tokenizer, self._pipe.tokenizer_2], text_encoder=[self._pipe.text_encoder, self._pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    def __call__(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        **d,
    ):
        conditioning, pooled = self.compel(prompt)
        negative_embed, negative_pooled = self.compel(negative_prompt)
        [conditioning, negative_embed] = self.compel.pad_conditioning_tensors_to_same_length([conditioning, negative_embed])
        return self._pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_embed,
            negative_pooled_prompt_embeds=negative_pooled,
            **d,
        )


def find_file(s):
    if not s:
        return None
    s = s.removesuffix('.safetensors')
    for i in [*Path(model_dir).glob('**/*.safetensors')] + [*Path(model_dir).glob('**/*.pt')] + [*Path(model_dir).glob('**/*.ckpt')]:
        if i.stem == s or i.name == s:
            return i
    raise Exception(f'找不到{s}！')


def pipeline0(model_type, path, vae_path) -> 超StableDiffusionKDiffusionPipeline | 超StableDiffusionXLPipeline | FluxPipeline:
    if model_type == 'sd':
        return 超StableDiffusionKDiffusionPipeline(path, vae_path)
    elif model_type == 'sdxl':
        return 超StableDiffusionXLPipeline(path, vae_path)
    elif model_type in ('flux.1s', 'flux.1d'):
        if model_type == 'flux.1s':
            repo = "black-forest-labs/FLUX.1-schnell"
        elif model_type == 'flux.1d':
            repo = "black-forest-labs/FLUX.1-dev"
        transformer = FluxTransformer2DModel.from_single_file(path, torch_dtype=torch.bfloat16).to('cuda')
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        text_encoder_2 = T5EncoderModel.from_pretrained(repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to('cuda')
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        pipe = FluxPipeline.from_pretrained(repo, transformer=None, text_encoder_2=None, torch_dtype=torch.bfloat16)
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2
        pipe.vae.enable_slicing()
        pipe = pipe.to('cuda')
        return pipe
    else:
        raise Exception(f'不认识模型类型{model_type}！')


def pipeline(model_type, path, vae_path, lora_path) -> 超StableDiffusionKDiffusionPipeline | 超StableDiffusionXLPipeline | FluxPipeline:
    for i in range(100):
        try:
            p = pipeline0(model_type, path, vae_path)   # 有时候网络不好
            break
        except Exception:
            if i == 99:
                raise
            time.sleep(30)
    if lora_path:
        print(lora_path)
        p._pipe.load_lora_weights(lora_path, adapter_name="Q")
        p._pipe.set_adapters(["Q"], adapter_weights=[1.0])
        p._pipe.fuse_lora(adapter_names=["Q"], lora_scale=1.0)
        p._pipe.unload_lora_weights()
    return p


_loaded_pipeline = None
def get_pipeline(model_type, model, vae, lora):
    global _loaded_pipeline
    if _loaded_pipeline and _loaded_pipeline[1] == (model_type, model, vae, lora):
        return _loaded_pipeline[0]
    torch.cuda.empty_cache()
    _loaded_pipeline = pipeline(model_type, find_file(model), find_file(vae), find_file(lora)), (model_type, model, vae, lora)
    torch.cuda.empty_cache()
    return _loaded_pipeline[0]


def txt2img(p: dict, 缓存=True, return_pil=False) -> list[bytes]:
    if 缓存:
        return _txt2img_缓存(p, return_pil)
    else:
        return _txt2img(p, return_pil)


@rimo_storage.cache.disk_cache(serialize='pickle')
def _txt2img_缓存(p: dict, return_pil=False) -> list[bytes]:
    return _txt2img(p, return_pil)


def _txt2img(p: dict, return_pil=False) -> list[bytes]:
    参数 = {}

    model_type = p.pop('model_type')

    参数['prompt'] = p.pop('prompt')
    参数['negative_prompt'] = p.pop('negative_prompt')
    参数['width'] = p.pop('width')
    参数['height'] = p.pop('height')
    参数['num_inference_steps'] = p.pop('steps')
    参数['guidance_scale'] = p.pop('cfg_scale')

    assert p.pop('sampler_name') == 'DPM++ 2M'
    assert p.pop('scheduler') == 'Karras'
    参数['use_karras_sigmas'] = True

    override_settings = p.pop('override_settings', None)

    batch_size = p.pop('batch_size')
    参数['prompt'] = [参数['prompt']] * batch_size
    参数['negative_prompt'] = [参数['negative_prompt']] * batch_size
    n_iter = p.pop('n_iter')
    seed = p.pop('seed')

    assert not p, f'剩下参数{p}不知道怎么转换……'

    pipe = get_pipeline(model_type, override_settings['sd_model_checkpoint'], override_settings['sd_vae'], override_settings.get('lora'))

    if model_type in ('flux.1s', 'flux.1d'):
        参数.pop('negative_prompt')
        参数.pop('use_karras_sigmas')
    if model_type == 'flux.1s':
        参数['num_inference_steps'] = 4

    res = []
    for i in range(n_iter):
        res.extend(pipe(
            **参数,
            generator=[torch.Generator(device='cuda').manual_seed(j) for j in range(seed + batch_size*i, seed + batch_size*(i+1))],
        ).images)
    torch.cuda.empty_cache()
    if return_pil:
        return res
    res_b = []
    for image in res:
        b = io.BytesIO()
        image.save(b, 'png')
        res_b.append(b.getvalue())
    return res_b


if __name__ == '__main__':
    pipe = 超StableDiffusionXLPipeline("R:/stable-diffusion-webui-master/models/Stable-diffusion/ConfusionXL4.0_fp16_vae.safetensors")
    from rimo_utils.计时 import 计时
    for i in range(3):
        with 计时():
            img = pipe(
                prompt=['1girl, white hair, twintails, standing'*10],
                negative_prompt=['worst quality, low quality'*10],
                guidance_scale=7,
                generator=torch.Generator(device='cuda').manual_seed(1),
                num_inference_steps=30,
                use_karras_sigmas=True,
            ).images[0]
    img.save(f"./q.png")
