# Deep Recursive Embedding

Deep Recursive Embedding (DRE) is a novel demensionality reduction method based on a generic deep embedding network (DEN) framework, which is able to learn a parametric mapping from high-dimensional space to low-dimensional space, guided by a recursive training strategy. DRE makes use of the latent data representations for boosted embedding performance.

## Installation

DRE can be installed with a simple PyPi command:

`pip install DRE`

## How to use DRE

DRE follows the form of Scikit-learn APIs, whose `fit_transform` function is for returning the embedding result and `fit` for the whole model:

```python
from DRE import DeepRecursiveEmbedding

dre = DeepRecursiveEmbedding()
y = dre.fit_transform(x)
```
