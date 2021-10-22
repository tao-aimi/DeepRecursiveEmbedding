import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from DRE import DeepRecursiveEmbedding


# Deep Recursive Embedding test code using MNIST/Fashion-MNIST datasets loaded with torchvision

transform_train = transforms.Compose([
    transforms.ToTensor(),
])
x_train = torchvision.datasets.MNIST(root='./datasets', train=False, transform=transform_train,
                                     download=True)
x_train_targets = np.int16(x_train.targets)
x_train = np.array(x_train.data).astype('float32')  # useful for GPU accelerating
# x_train = x_train.reshape(x_train.shape[0], -1)

dre = DeepRecursiveEmbedding(dre_type='conv',
                             n_pre_epochs=100,
                             num_recursive_tsne_epochs=50,
                             num_recursive_umap_epochs=100,
                             learning_rate=1e-3,
                             batch_size=2500,
                             random_shuffle=False,  # for plotting with labels, set to 'False'
                             save_directory='./',
                             )
# dre.labels = x_train_targets

y = dre.fit_transform(x_train)

# Plot the result:
labels = x_train_targets
colors = ['darkorange', 'deepskyblue', 'gold', 'lime', 'k', 'darkviolet', 'peru', 'olive',
               'midnightblue',
               'palevioletred']
cmap = matplotlib.colors.ListedColormap(colors[::-1])
fig = plt.figure(figsize=(10, 10))
fig.patch.set_facecolor('#303030')
scatter = plt.scatter(y[:, 0], y[:, 1], s=0.1, cmap=cmap, c=labels, alpha=0.5)
legend1 = plt.legend(*scatter.legend_elements(), title="Classes", loc='upper right')
plt.axis('equal')
plt.axis("off")
plt.show()

# Save the model (default selected dir: ./):
dre.save_model(save_dir='./')  # The model is saved in '<selected dir>/DRE_model_checkpoint/'

