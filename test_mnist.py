import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

import os
import time
import math
import matplotlib
import matplotlib.pyplot as plt

from DRE import DeepRecursiveEmbedding


# Deep Recursive Embedding test code using MNIST/Fashion-MNIST datasets loaded with torchvision

transform_train = transforms.Compose([
    transforms.ToTensor(),
])
x_train = torchvision.datasets.FashionMNIST(root='./datasets', train=True, transform=transform_train,
                                     download=True)
x_train_targets = np.int16(x_train.targets)
x_train = np.array(x_train.data).astype('float32')  # useful for GPU accelerating
x_train = x_train / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)

dre = DeepRecursiveEmbedding()

y = dre.fit_transform(x_train)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_aspect('equal')

colors = ['darkorange', 'deepskyblue', 'gold', 'lime', 'k', 'darkviolet', 'peru', 'olive', 'midnightblue',
              'palevioletred']
cmap = matplotlib.colors.ListedColormap(colors[::-1])
# y_mean = cluster_mean(y_test, x_test_targets)
scatter1 = ax.scatter(y[:, 0], y[:, 1], s=0.1, cmap=cmap, c=x_train_targets[:], alpha=0.7)
plt.axis('off')
# plt.savefig('fmnist_near%d-middle%d.png' % (spacemap.chi_knn, spacemap.semi_knn))
plt.show()



