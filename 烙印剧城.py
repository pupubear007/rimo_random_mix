import sys
import json
import time
import random
import hashlib

import numpy as np
from bayes_opt import BayesianOptimization
from safetensors.numpy import load_file, save_file

sys.path.append('R:/e')
from common import 上网, 服务器地址
from 评测多标签 import 评测模型


from safetensors.numpy import load_file, save_file
import numpy as np


rd = random.Random(int(time.time()))


模型文件夹 = 'R:/stable-diffusion-webui-master/models/Stable-diffusion'

所有模型 = [
    'AOM3A1',
    'Counterfeit-V2.2',
    'Counterfeit-V3.0_fp16',
    'anyloraCheckpoint_novaeFp16',
    'bluePencil_v10',
    'calicomix_v75',
    'cetusMix_v4',
    'cuteyukimixAdorable_specialchapter',
    'sweetMix_v22Flat',
    'petitcutie_v15',
    'kaywaii_v70',
    'kaywaii_v90',
    'rimochan_random_mix_1.1',
    'superInvincibleAnd_v2',
    'sakuramochimix_v10',
    'sweetfruit_melon.safetensors_v1.0',
    'AnythingV5Ink_ink',
    'rabbit_v7',
    'rainbowsweets_v20',
    'himawarimix_v100',
    'koji_v21',
    'yetanotheranimemodel_v20',
    'irismix_v90',
    'theWondermix_v12',
]


def 融合识别(s: str, p: int = 4) -> str:
    nm = {
        'x': 'model.diffusion_model.input_blocks.',
        'y': 'model.diffusion_model.middle_block.',
        'z': 'model.diffusion_model.output_blocks.',
    }
    for k, v in nm.items():
        if s.startswith(v):
            n = int(s.removeprefix(v).split('.')[0])
            return f'{k}_{n//p}'
    return 'r'


def 烙(**kw):
    新模型 = {}
    for k in 所有层:
        识别k = 融合识别(k)
        aw = 1
        for i, _ in enumerate(其他模型):
            aw -= kw[f'{i}_{识别k}']
        新模型[k] = a[k].astype(np.float32) * aw
        for i, b in enumerate(其他模型):
            新模型[k] += b[k].astype(np.float32) * kw[f'{i}_{识别k}']
    文件名 = 名字(kw)
    save_file(新模型, f'{模型文件夹}/{文件名}.safetensors')
    上网(f'{服务器地址}/sdapi/v1/refresh-checkpoints', method='post')
    结果 = 评测模型(文件名, 'blessed2.vae.safetensors', 32, n_iter=80, use_tqdm=False, savedata=False, seed=seed, tags_seed=tags_seed, 计算相似度=False)
    m = []
    for dd in 结果:
        m.extend(dd['分数'])
    mm = np.array(m)
    acc = (mm > 0.001).sum() / len(mm.flatten())
    记录.append({
        '文件名': 文件名,
        'acc': acc,
    })
    print(文件名, acc, mm.shape)
    with open(记录文件名, 'w', encoding='utf8') as f:
        json.dump(记录, f, indent=2, ensure_ascii=False)
    return acc


def 名字(kw: dict):
    s = sorted(kw.items())
    md5 = hashlib.md5(str(''.join(f'{k}{v:.2f}' for k, v in s)).encode()).hexdigest()
    return f'Theater_{当前标记}_{md5[:8]}'


当前模型 = 'rimochan_random_mix_3.2'


标记 = rd.randint(0, 10000)

for i in range(100):
    当前标记 = f'{标记}_{i}'
    记录文件名 = f'记录_烙印剧城_{当前标记}_{int(time.time())}.txt'
    记录 = []
    其他模型名 = rd.sample(所有模型, 3)

    记录.append(['融合', 当前模型, 其他模型名])
    print('融合', 当前模型, 其他模型名)

    a = load_file(f'{模型文件夹}/{当前模型}.safetensors')
    其他模型 = [load_file(f'{模型文件夹}/{i}.safetensors') for i in 其他模型名]

    所有层 = set(a)
    for b in 其他模型:
        所有层 &= set(b)

    识别结果 = {融合识别(i) for i in 所有层}

    所有参数 = []
    for i, b in enumerate(其他模型):
        for j in 识别结果:
            所有参数.append(f'{i}_{j}')

    print('所有参数为:', 所有参数)

    optimizer = BayesianOptimization(
        f=烙,
        pbounds={i: (-0.2, 0.4) for i in 所有参数},
        random_state=666,
    )

    seed = rd.randint(1000, 9000)
    tags_seed=random.randint(1000, 9000)

    optimizer.probe(
        params={i: 0 for i in 所有参数},
    )

    optimizer.maximize(
        init_points=4,
        n_iter=35,
    )

    当前模型 = 名字(optimizer.max['params'])
