# Deep Recursive Embedding

Deep Recursive Embedding (DRE) is a novel demensionality reduction method based on a generic deep embedding network (DEN) framework, which is able to learn a parametric mapping from high-dimensional space to low-dimensional space, guided by a recursive training strategy. DRE makes use of the latent data representations for boosted embedding performance.

Lab github DRE page:
[Tao Lab](https://github.com/tao-aimi/DeepRecursiveEmbedding)

Maintainer's github DRE page:
[Xinrui Zu](https://github.com/zuxinrui/DeepRecursiveEmbedding)

## MNIST embedding result

![gif](/images/MNIST-conv-2.gif)

## Installation

DRE can be installed with a simple PyPi command:

`pip install DRE`

The pre-requests of DRE are:

`numpy >= 1.19`
`scikit-learn >= 0.16`
`matplotlib`
`numba >= 0.34`
`torch >= 1.0`

## How to use DRE

DRE follows the form of `Scikit-learn` APIs, whose `fit_transform` function is for returning the embedding result and `fit` for the whole model:

```python
from DRE import DeepRecursiveEmbedding

dre = DeepRecursiveEmbedding()
# return the embedding result:
y = dre.fit_transform(x)
# or return the whole model:
dre.fit(x)
```
Copy and run `test_mnist.py` or `test_mnist.ipynb` to check the embedding procedure of MNIST dataset.

## Citation
Z. Zhou, X. Zu, Y. Wang, B. P. F. Lelieveldt and Q. Tao, "Deep Recursive Embedding for High-Dimensional Data," in IEEE Transactions on Visualization and Computer Graphics, vol. 28, no. 2, pp. 1237-1248, 1 Feb. 2022, doi: 10.1109/TVCG.2021.3122388.

@ARTICLE{DRE2022,
  author={Zhou, Zixia and Zu, Xinrui and Wang, Yuanyuan and Lelieveldt, Boudewijn P. F. and Tao, Qian},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={Deep Recursive Embedding for High-Dimensional Data}, 
  year={2022},
  volume={28},
  number={2},
  pages={1237-1248},
  doi={10.1109/TVCG.2021.3122388}
  }


## Link
https://ieeexplore.ieee.org/document/9585419

## 
