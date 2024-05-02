# 莉沫酱的随机融合模型！

事情是这样的，最近我测试了不少网上的stable diffusion的动漫风格模型。

其中有1些模型的效果还不错，不过它们各自都有些缺点，于是我就想，那我来做1个天衣无缝的模型好了！

原理是这样的，有2个假设: 

- 合并模型的权重不会引入额外的过拟合。

- 在符合语义方面表现得更好的模型在其他方面也能表现得更好。

嘛，直觉上是这样，第1个假设应该是对的，第2个……我不好说。要是问我为什么我就回答「有人托梦给我」。

总之，这样1来，我们只要对于每个层，选择它是由哪几个模型的该层以什么样的权重融合，然后在所有权重参数的空间里做搜索，最终让准确度最高就可以了。


## 效果

测指标的仓库是这个: <https://github.com/RimoChan/stable-diffusion-anime-tag-benchmark>

具体的指标计算方式和详细的指标，感兴趣的话可以点到里面去看看，这里就只说结论，是这样——

标签准确度，相比ACG模型的平均值提升37%，并且在各个方向均有8%~151%的提升。

这个比例随着prompts的数量上升，在32标签时提升56%，高于所有典型模型的准确度。

各个方向的情况是这样，数值为准确度:

|            |   CXL2.0 |   平均值 | 偏差     |
|:-----------|---------:|---------:|:---------|
| 艺术破格   |    0.353 |    0.141 | +150.35% |
| 人物      |    0.846 |    0.608 | +39.14%  |
| 人文景观 |    0.947 |    0.859 | +10.24%  |
| 构图      |    0.689 |    0.455 | +51.43%  |
| 物品       |    0.94  |    0.791 | +18.84%  |
| 自然景观    |    0.982 |    0.905 | +8.51%   |
| 限制级 |    0.679 |    0.367 | +85.01%  |
| 总体       |    0.829 |    0.603 | +37.48%  |

生成图片的具体效果在这个文档里，有1些可爱的测试图片，有兴趣的话进去可以看1下: <https://c.librian.net/>


## 模型下载

Github的LFS超过1G居然要收钱！所以我就把模型传到Civitai了，下载的链接在这里:

普通版: <https://civitai.com/models/249129>

XL版: <https://civitai.com/models/358055>


## 原理

我们前面说要直接搜出1个指标最高的模型嘛，所以做法是这样:

假设我们手里有2个模型a和b，它们分别有3层，即ax、ay、az和bx、by、bz。

那么我们想要得到1个新的模型c，它也有3层cx、cy、cz，那么可以这样得到c: 

```python
cx = w1 * ax + (1-w1) * bx
cy = w2 * ay + (1-w2) * by
cz = w3 * az + (1-w3) * bz
```

其中，ax、ay、az和bx、by、bz是已知的，w1、w2、w3是3个未知数。

我们将w1、w2、w3作为贝叶斯搜索的参数，这样1来，每进行1轮贝叶斯搜索，我们就能得到1个确定的模型c。

有了模型c，我们就可以对c测指标<sub>(也是用上面那个仓库)</sub>，然后将测出来的指标送回贝叶斯搜索的奖励函数，让它进行下1轮搜索。

这样1来，就可以不停地搜索，然后你就等着，等到指标基本不涨<sub>(大概需要几天时间)</sub>，就可以得到1个生成效果不错的模型了。

当然实际代码里会复杂1些，比如2个模型可以扩展到n个，就不具体说它们了，细节可以参考[烙印融合.py](烙印融合.py)和[烙印剧城.py](烙印剧城.py)这2个代码。


## 结束

好，就这样，大家88，我要回去和`1girl`亲热了！

还有我突然想起来天衣无缝，那天衣其实是乳胶衣吧！
