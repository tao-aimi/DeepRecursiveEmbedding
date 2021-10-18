# Deep Recursive Embedding

Deep Recursive Embedding (DRE) is a novel demensionality reduction method based on a generic deep embedding network (DEN) framework, which is able to learn a parametric mapping from high-dimensional space to low-dimensional space, guided by a recursive training strategy. DRE makes use of the latent data representations for boosted embedding performance.

Lab github DRE page:
[Tao Lab](https://github.com/tao-aimi/DeepRecursiveEmbedding)

Maintainer's github DRE page:
[Xinrui Zu](https://github.com/zuxinrui/DeepRecursiveEmbedding)

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

DRE follows the form of Scikit-learn APIs, whose `fit_transform` function is for returning the embedding result and `fit` for the whole model:

```python
from DRE import DeepRecursiveEmbedding

dre = DeepRecursiveEmbedding()
y = dre.fit_transform(x)
```
Run `test_MNIST.py` to check the embedding procedure of MNIST dataset.

## 
