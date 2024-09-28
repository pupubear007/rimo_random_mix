import os
import json
import time
import random
import hashlib

import torch
import numpy as np
from bayes_opt import BayesianOptimization
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from benchmarker.评测多标签改 import 评测模型
from tqdm import tqdm

from prometheus_client import start_http_server, Gauge


start_http_server(10721)
rd = random.Random(int(time.time()))
acc打点 = Gauge('acc', 'acc')


模型文件夹 = 'R:/stable-diffusion-webui-master/models/Stable-diffusion'


vae = 'sdxl_vae_0.9.safetensors'
所有模型 = [
    'aingdiffusionXL_v11',
    'animagineXLV3_v30',
    'aniease_v24',
    'animeIllustDiffusion_v052',
    'counterfeitxl_v25',
    'himawarimix_xlV13',
    'reproductionSDXL_2v12',
    'kohakuXLBeta_beta7',
    'kohakuXLDelta_rev1',
    'baxlBartstylexlBlueArchiveFlatCelluloid_xlv2',
]
融合模型个数 = 3
n_iter = 44
当前模型 = 'aniease_v24'


def 融合识别(s: str) -> str:
    nm = {
        'x': 'model.diffusion_model.input_blocks.',
        'y': 'model.diffusion_model.middle_block.',
        'z': 'model.diffusion_model.output_blocks.',
    }
    for k, v in nm.items():
        if s.startswith(v):
            n = int(s.removeprefix(v).split('.')[0])
            if k == 'x':
                for z, s in enumerate(((0, 1, 2, 3, 4, 5, 6), (7,), (8,))):
                    if n in s:
                        真n = z
            elif k == 'y':
                真n = 0
            elif k == 'z':
                for z, s in enumerate(((0,), (1,), (2,), (3, 4, 5, 6, 7, 8))):
                    if n in s:
                        真n = z
            else:
                raise Exception('啊？')
            return f'{k}_{真n}'
    return 'r'


def 烙(**kw):
    新模型 = {}
    branded_from: dict[str, dict[str, float]] = {}
    for k in 所有层:
        识别k = 融合识别(k)
        aw = 1
        for i, _ in enumerate(其他模型):
            aw -= kw[f'{i}_{识别k}']
        新模型[k] = torch.from_numpy(a[k]).cuda().to(torch.float32) * aw
        branded_from[k] = {kk: vv * aw for kk, vv in 原branded_from.get(k, {当前模型: 1}).items()}
        for i, b in enumerate(其他模型):
            bw = kw[f'{i}_{识别k}']
            新模型[k] += torch.from_numpy(b[k]).cuda().to(torch.float32) * bw
            branded_from[k][其他模型名[i]] = branded_from[k].get(其他模型名[i], 0) + bw
        新模型[k] = 新模型[k].to(torch.float16).cpu().numpy()
    文件名 = 名字(kw)
    save_file(新模型, f'{模型文件夹}/{文件名}.safetensors', metadata={'branded_from': json.dumps(branded_from, ensure_ascii=False, default=float)})
    del 新模型
    生成测试样例 = all([i==0 for i in kw.values()])
    结果 = 评测模型(文件名, vae, 32, n_iter=100, use_tqdm=True, tags_seed=tags_seed, 生成测试样例=生成测试样例)
    m = []
    tm = []
    for dd in 结果:
        m.extend(dd['分数'])
        for k, v in dd['预测标签'].items():
            tm += v
    mm = np.array(m)
    acc = (mm > 0.001).sum() / len(mm.flatten())
    记录.append({
        '文件名': 文件名,
        'acc': acc,
    })
    acc打点.set(acc)
    惩罚 = sum([i for i in kw.values() if i < 0]) * (-0.005)
    print(文件名, f'acc={acc}', f'惩罚={惩罚}', mm.shape)
    with open(记录文件名, 'w', encoding='utf8') as f:
        json.dump(记录, f, indent=2, ensure_ascii=False)
    return acc - 惩罚


def 名字(kw: dict):
    s = sorted(kw.items())
    md5 = hashlib.md5(str(''.join(f'{k}{v:.2f}' for k, v in s)).encode()).hexdigest()
    return f'Theater_{当前标记}_{md5[:8]}'


标记 = rd.randint(0, 10000)
队 = []
for i in range(100):
    t = 所有模型.copy()
    random.shuffle(t)
    队 += t
for i in range(100):
    当前标记 = f'{标记}_{i}'
    记录文件名 = f'记录/记录_烙印剧城_{当前标记}_{int(time.time())}.txt'
    记录 = []
    其他模型名 = 队[:融合模型个数]
    队 = 队[融合模型个数:]
    记录.append(['融合', 当前模型, 其他模型名])
    print('融合', 当前模型, 其他模型名)

    a = load_file(f'{模型文件夹}/{当前模型}.safetensors')
    原branded_from = json.loads((safe_open(f'{模型文件夹}/{当前模型}.safetensors', framework='np').metadata() or {}).get('branded_from', '{}'))
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
        pbounds={i: (-0.12, 0.5) for i in 所有参数},
        random_state=666,
    )

    tags_seed = rd.randint(1000, 9000)

    optimizer.probe(
        params={i: 0 for i in 所有参数},
    )

    optimizer.maximize(
        init_points=6,
        n_iter=n_iter,
    )

    for i in sorted(optimizer.res, key=lambda x: -x['target'])[3:]:
        try:
            os.remove(f'{模型文件夹}/{名字(i["params"])}.safetensors')
        except Exception:
            None
    当前模型 = 名字(optimizer.max['params'])
