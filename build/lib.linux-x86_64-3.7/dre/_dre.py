import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

import os
import time
import math
from tqdm import tqdm
from math import ceil
import matplotlib.pyplot as plt
import matplotlib

from ._models import *
from ._utils_dre import *


# os.environ['MKL_DEBUG_CPU_TYPE'] = '5'  # 3900x


class DeepRecursiveEmbedding:
    """
    	****** Deep Recursive Embedding implementation (PyTorch). ******
		Author: Xinrui Zu
		Version: 1.0
		Date: 2021/10/10

		Original Paper:
		Authors: Zixia Zhou, Xinrui Zu, Yuanyuan Wang, Boudewijn P.F. Lelieveldt, Qian Tao

        n_components:
        The embedding dimensionality.

        n_pre_epochs:
        The number of epochs before the recursive steps.

        n_recursive_epochs:
        The number of epochs in each recursive step. (default total steps: 300+100*4)

        dre_type (default 'fc'):
        The default type of the network. options: ['fc', 'conv']

        learning_rate:
        The learning rate of the optimization.

        tsne_perplexity (default: 30.0):
        The parameter defininig the normalization of t-SNE Pij.

        umap_n_neighbors (default: 15):
        The number of neighbors when calculating Pij in the last UMAP-like recursive step.

        min_dist:
        The minimum distance between the embedding points. Used to form Qij.

        batch_size (default: 2500):
        The batch size of the training procedure.

        rebatching_epochs:
		The number of epochs when the minibatches are fixed in the total dataset.

		data_dim:
		The dimensionality of the input data (when using fully-connected DRE).

		data_dim_conv:
		The dimensionality of the input tensor (when using convolutional DRE).
		Example: Fashion-MNIST: [3,32,32]
    """
    def __init__(self,
                 n_components=2,
                 n_pre_epochs=300,
                 n_recursive_epochs=150,
                 dre_type='fc',  # 'fc' or 'conv'
                 learning_rate=5.0 * 1e-5,
                 tsne_perplexity=30.0,
                 umap_n_neighbors=15,
                 min_dist=0.001,
                 batch_size=2500,
                 rebatching_epochs=1e4,
                 data_dim=0,
                 data_dim_conv=[],
                 ):
        self.dre_type = dre_type
        self.learn_from_exist = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lowest_loss = 10000.0
        self.num_pre_epochs = n_pre_epochs
        self.num_recursive_epochs = n_recursive_epochs
        self.num_epochs = self.num_pre_epochs + 4*self.num_recursive_epochs
        self.batch_size = batch_size

        self.perplexity = tsne_perplexity
        self.umap_knn = umap_n_neighbors
        self.min_dist = min_dist
        self.a, self.b = find_ab_params(1, self.min_dist)

        self.lr = learning_rate

        self.rolling_num = rebatching_epochs
        self.data_dim = data_dim
        self.data_dim_conv = data_dim_conv

        self.P = []
        self.data = 0
        self.num_batch = 0
        self.embedding = 0

        self.loss_plot_train = []
        self.loss_plot_val = []

        # self.num_batch = int(self.data.shape[0] / self.batch_size)
        # self.n = self.num_batch * self.batch_size

        # self.start_time = time.time()
        if not os.path.isdir('checkpoint'):
        	os.mkdir('checkpoint')

        print('<================ Building model ================>')

        if self.dre_type == 'conv':
            self.net = DREConvSmall(*self.data_dim_conv)
        elif self.dre_type == 'fc':
            self.net = DRE(self.data_dim)
        else:
            raise TypeError('[DRE] DRE type must be conv or normal.')

        self.net = self.net.to(self.device)
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        print('<================ Model loaded ================>')

        # Loss function and optimization method:

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)  # 5.0*(1e-5) for the original code
        # self.optimizer = optim.SGD(net.parameters(), lr=1e-4,
        #                       momentum=0.9, weight_decay=5e-4)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1500)

    def calculate_p_matrix(self, step):
        # if self.data == 0:
        #     raise TypeError('[DRE] Input the data first.')
        ran_num = np.random.randint(2 ** 16 - 1)
        np.random.seed(ran_num)
        np.random.shuffle(self.data)
        return self._calculate_p_matrix(step)  # fill in 'pre', 're1', 're2', 're3', 're_umap'

    def _calculate_p_matrix(self, step):
        print('building P matrix...')
        self.P = []
        self.net.eval()
        with torch.no_grad():
            if step == 'pre':
                for i in range(self.num_batch):
                    inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, inputs, p_type='tsne')
                    self.P.append(P1)
                    print('[DRE] P length: ', len(self.P))
            if step == 're1':
                for i in range(self.num_batch):
                    if self.dre_type == 'conv':
                        inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2)
                    elif self.dre_type == 'fc':
                        inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                        inputs = torch.from_numpy(inputs)
                    else:
                        raise TypeError('DRE type must be conv or fc.')
                    low_dim_data, _, _, _ = self.net(inputs)
                    low_dim_data = np.array(low_dim_data.cpu())
                    print('[DRE] recursive step 1 data shape: ', low_dim_data.shape)
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, low_dim_data, p_type='tsne')
                    self.P.append(P1)
                    print('[DRE] P length: ', len(self.P))
            if step == 're2':
                for i in range(self.num_batch):
                    if self.dre_type == 'conv':
                        inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2)
                    elif self.dre_type == 'fc':
                        inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                        inputs = torch.from_numpy(inputs)
                    else:
                        raise TypeError('DRE type must be conv or fc.')
                    _, low_dim_data, _, _ = self.net(inputs)
                    low_dim_data = np.array(low_dim_data.cpu())
                    print('[DRE] recursive step 2 data shape: ', low_dim_data.shape)
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, low_dim_data, p_type='tsne')
                    self.P.append(P1)
                    print('[DRE] P length: ', len(self.P))
            if step == 're3':
                for i in range(self.num_batch):
                    if self.dre_type == 'conv':
                        inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2)
                    elif self.dre_type == 'fc':
                        inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                        inputs = torch.from_numpy(inputs)
                    else:
                        raise TypeError('DRE type must be conv or fc.')
                    _, _, low_dim_data, _ = self.net(inputs)
                    low_dim_data = np.array(low_dim_data.cpu())
                    print('[DRE] recursive step 3 data shape: ', low_dim_data.shape)
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, low_dim_data, p_type='tsne')
                    self.P.append(P1)
                    print('[DRE] P length: ', len(self.P))
            if step == 're_umap':
                for i in range(self.num_batch):
                    if self.dre_type == 'conv':
                        inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2)
                    elif self.dre_type == 'fc':
                        inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                        inputs = torch.from_numpy(inputs)
                    else:
                        raise TypeError('DRE type must be conv or fc.')
                    _, _, low_dim_data, _ = self.net(inputs)
                    low_dim_data = np.array(low_dim_data.cpu())
                    print('[DRE] recursive step 3 data shape: ', low_dim_data.shape)
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, low_dim_data, p_type='umap')
                    self.P.append(P1)
                    print('[DRE] P length: ', len(self.P))

        return self.P

    def _train(self, epoch, step):
        print('\nEpoch: %d' % epoch)
        self.net.train()  # train mode
        loss_all = 0
        with tqdm(total=self.data.shape[0], desc=f'[DRE] Training.. Epoch {epoch + 1}/{self.num_epochs}', unit='img', colour='blue') as pbar:
            for i in range(self.num_batch):
                # Load the packed data:
                tar = self.P[i]
                inputs, targets = torch.from_numpy(self.data[i*self.batch_size:(i+1)*self.batch_size]).type(torch.FloatTensor).to(self.device),\
                                  torch.from_numpy(tar).type(torch.FloatTensor).to(self.device)  # 如果是long会全变成0

                inputs = inputs.reshape(inputs.shape[0], -1)
                self.optimizer.zero_grad()
                _, _, _, outputs = self.net(inputs)

                loss = loss_function(targets, outputs, self.a, self.b, type=step)
                # print(loss)
                if str(loss.item()) == str(np.nan):
                    print('[DRE] detect nan in loss function, skip this iter')
                    continue
                loss.backward()
                loss_all += loss.item()
                # losses[divide_number*i+j] = loss
                self.optimizer.step()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(targets.shape[0])
        return loss_all / self.num_batch

    def _train_conv(self, epoch, step):
        print('\nEpoch: %d' % epoch)
        self.net.train()  # train mode
        loss_all = 0
        with tqdm(total=self.data.shape[0], desc=f'[DRE] Training.. Epoch {epoch + 1}/{self.num_epochs}', unit='img', colour='blue') as pbar:
            for i in range(self.num_batch):
                # Load the packed data:
                tar = self.P[i]
                inputs, targets = torch.from_numpy(self.data[i*self.batch_size:(i+1)*self.batch_size])\
                                      .type(torch.FloatTensor).permute(0, 3, 1, 2).to(self.device),\
                                  torch.from_numpy(tar).type(torch.FloatTensor).to(self.device)  # 如果是long会全变成0
                self.optimizer.zero_grad()
                _, _, _, outputs = self.net(inputs)

                loss = loss_function(targets, outputs, self.a, self.b, type=step)
                if str(loss.item()) == str(np.nan):
                    print('[DRE] detect nan in loss function, skip this iter')
                    continue
                loss.backward()
                loss_all += loss.item()
                # losses[divide_number*i+j] = loss
                self.optimizer.step()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(targets.shape[0])
        return loss_all / self.num_batch

    def _validation(self, epoch, step):
        self.net.eval()  # train mode
        loss_all = 0
        with torch.no_grad():
            with tqdm(total=self.data.shape[0], desc=f'[DRE] Validating.. Epoch {epoch + 1}/{self.num_epochs}', unit='img',
                      colour='green') as pbar:
                for i in range(self.num_batch):
                    # Load the packed data:
                    tar = self.P[i]
                    inputs, targets = torch.from_numpy(self.data[i*self.batch_size:(i+1)*self.batch_size]).type(torch.FloatTensor).to(self.device), \
                                      torch.from_numpy(tar).type(torch.FloatTensor).to(self.device)
                    inputs = inputs.reshape(inputs.shape[0], -1)
                    _, _, _, outputs = self.net(inputs)
                    # loss = kl_divergence_bayes(outputs, targets, 1, knn_bayes)
                    loss = loss_function(targets, outputs, self.a, self.b, type=step)
                    loss_all += loss
                    # loss_aver = loss_all / (i+1)
                    pbar.set_postfix(**{'loss (batch)': loss_all / self.num_batch})
                    pbar.update(targets.shape[0])
        # Save checkpoint.
        if float(loss_all / self.num_batch) < self.lowest_loss:
            print('[DRE] Best accuracy, saving the weights...')
            state = {
                'net': self.net.state_dict(),
                'loss': loss.item(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            # Remember to rename the saved model corresponding to the hyper-parameters:
            torch.save(state, './checkpoint/ckpt_DRE_multi_fresh.pth')
            lowest_loss = float(loss_all / self.num_batch)
        return loss_all / self.num_batch

    def _validation_conv(self, epoch, step):
        self.net.eval()  # train mode
        loss_all = 0
        with torch.no_grad():
            with tqdm(total=self.data.shape[0], desc=f'[DRE] Validating.. Epoch {epoch + 1}/{self.num_epochs}', unit='img',
                      colour='green') as pbar:
                for i in range(self.num_batch):
                    # Load the packed data:
                    tar = self.P[i]
                    inputs, targets = torch.from_numpy(self.data[i*self.batch_size:(i+1)*self.batch_size])\
                                          .type(torch.FloatTensor).permute(0, 3, 1, 2).to(self.device), \
                                      torch.from_numpy(tar).type(torch.FloatTensor).to(self.device)  # 如果是long会全变成0
                    _, _, _, outputs = self.net(inputs)
                    # loss = kl_divergence_bayes(outputs, targets, 1, knn_bayes)
                    loss = loss_function(targets, outputs, self.a, self.b, type=step)
                    loss_all += loss
                    # loss_aver = loss_all / (i+1)
                    pbar.set_postfix(**{'loss (batch)': loss_all / self.num_batch})
                    pbar.update(targets.shape[0])
        # Save checkpoint.
        if float(loss_all / self.num_batch) < self.lowest_loss:
            print('[DRE] Best accuracy, saving the weights...')
            state = {
                'net': self.net.state_dict(),
                'loss': loss.item(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            # Remember to rename the saved model corresponding to the hyper-parameters:
            torch.save(state, './checkpoint/ckpt_DRE_multi_fresh.pth')
            lowest_loss = float(loss_all / self.num_batch)
        return loss_all / self.num_batch

    def plot(self, epoch, step):
        self._plot(epoch, step)

    def _plot(self, epoch, step, fig_size='normal'):
        self.net.eval()
        with torch.no_grad():
            for i in range(self.num_batch):
                if self.dre_type == 'conv':
                    inputs = torch.from_numpy(self.data[i * self.batch_size:(i + 1) * self.batch_size]).type(
                                                    torch.FloatTensor)  # conv
                elif self.dre_type == 'fc':
                    inputs = torch.from_numpy(self.data[i * self.batch_size:(i + 1) * self.batch_size])\
                        .type(torch.FloatTensor)
                else:
                    raise TypeError('DRE type must be conv or fc.')

                if i == 0:
                    _, _, _, Y = self.net(inputs)
                    Y = Y.cpu()
                else:
                    _, _, _, y_test = self.net(inputs)
                    Y = np.concatenate((Y, y_test.cpu()), axis=0)
        torch.cuda.empty_cache()
        plt.figure(figsize=(10, 10))
        if fig_size == 'fixed':
            plt.xlim([-500, 500])
            plt.ylim([-500, 500])
        scatter = plt.scatter(Y[:, 0], Y[:, 1], s=1, c='darkorange', alpha=0.7)
        # legend1 = plt.legend(*scatter.legend_elements(), title="Classes", loc='upper right')
        plt.title('Epoch = %d, Loss = %f' % (epoch, self.loss_score_train))
        plt.tight_layout()
        # plt.scatter(Y[:, 0], Y[:, 1], s=1, c=targets)
        # plt.scatter(Y[:, 0], Y[:, 1], s=1, c=plt.cm.cubehelix(0.1 * targets))
        plt.savefig("./DRE_labeled_%s_%s.png" % (time.asctime(time.localtime(time.time())), step))
        if step == 're_umap':
            return Y

    def _fit_embedding(self):
        start_time = time.time()
        print('[DRE] start------->  time: ', time.ctime(time.time()))
        recursive_step = 'pre'
        for epoch in range(self.num_epochs):

            if epoch == 0:
                self.P = self.calculate_p_matrix(recursive_step)

            if epoch % self.rolling_num == 0 and epoch != 0 and epoch not in \
                    np.int16(self.num_pre_epochs + np.int16([0, 1, 2, 3])*self.num_recursive_epochs):
                self.P = self.calculate_p_matrix(recursive_step)

            if epoch == self.num_pre_epochs:  # 300
                # save the model:
                state = {
                    'net': self.net.state_dict(),
                    'loss': self.loss_score_train,
                    'epoch': epoch,
                }
                torch.save(state, './checkpoint/DRE_{}.pth'.format(recursive_step))

                self.plot(epoch, recursive_step)

                recursive_step = 're1'
                # calculate the new P matrix:
                self.P = self.calculate_p_matrix(recursive_step)

            if epoch == self.num_pre_epochs + self.num_recursive_epochs:  # 400
                # save the model:
                state = {
                    'net': self.net.state_dict(),
                    'loss': self.loss_score_train,
                    'epoch': epoch,
                }
                torch.save(state, './checkpoint/DRE_{}.pth'.format(recursive_step))

                self.plot(epoch, recursive_step)

                recursive_step = 're2'
                # calculate the new P matrix:
                self.P = self.calculate_p_matrix(recursive_step)

            if epoch == self.num_pre_epochs + 2*self.num_recursive_epochs:  # 500
                # save the model:
                state = {
                    'net': self.net.state_dict(),
                    'loss': self.loss_score_train,
                    'epoch': epoch,
                }
                torch.save(state, './checkpoint/DRE_{}.pth'.format(recursive_step))

                self.plot(epoch, recursive_step)

                recursive_step = 're3'
                # calculate the new P matrix:
                self.P = self.calculate_p_matrix(recursive_step)

            if epoch == self.num_pre_epochs + 3*self.num_recursive_epochs:  # 600
                # save the model:
                state = {
                    'net': self.net.state_dict(),
                    'loss': self.loss_score_train,
                    'epoch': epoch,
                }
                torch.save(state, './checkpoint/DRE_{}.pth'.format(recursive_step))

                self.plot(epoch, recursive_step)

                recursive_step = 're_umap'
                # calculate the new P matrix:
                self.P = self.calculate_p_matrix(recursive_step)

            # <==================== train ====================>

            if self.dre_type == 'conv':
                self.loss_score_train = self._train_conv(epoch, recursive_step)
            elif self.dre_type == 'fc':
                self.loss_score_train = self._train(epoch, recursive_step)
            else:
                raise TypeError('DRE type should be conv or fc.')

            # <==================== validate ====================>

            if (epoch + 1) % 10 == 0:
                if self.dre_type == 'conv':
                    self.loss_score_val = self._validation_conv(epoch, recursive_step).cpu()
                elif self.dre_type == 'fc':
                    self.loss_score_val = self._validation(epoch, recursive_step).cpu()
                else:
                    raise TypeError('DRE type should be conv or fc.')
                self.loss_plot_val.append(self.loss_score_val)
            # scheduler.step()
            self.loss_plot_train.append(self.loss_score_train)

        end_time = time.time()
        duration = end_time - start_time
        print('[DRE] training time: ', duration)
        state = {
            'net': self.net.state_dict(),
            'loss': self.loss_score_train,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/DRE_{}.pth'.format(recursive_step))

        print('[DRE] ------->complete.  time: ', time.ctime(time.time()))

        # plot validation loss
        plt.clf()
        # plt.style.use(style='Solarize_Light2')
        plt.plot(self.loss_plot_val, color='b')
        plt.title('Loss in the iteration (validation). Epochs = %d, Learning rate: %f' % (self.num_epochs, self.lr))
        plt.xlabel('epochs / 10')
        plt.ylabel('loss')
        plt.savefig("./loss_val_{}.png".format(recursive_step))

        # plot training loss
        plt.clf()
        plt.plot(self.loss_plot_train, color='b')
        plt.title('Loss in the iteration (training). Epochs = %d, Learning rate: %f' % (self.num_epochs, self.lr))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig("./loss_train_{}.png".format(recursive_step))

        self.plot(epoch, recursive_step)

    def fit_transform(self, x):
        start_time = time.time()
        self.data = x
        self.num_batch = ceil(self.data.shape[0] / self.batch_size)
        self._fit_embedding()
        end_time = time.time()
        print('fitting time: {}s'.format(end_time - start_time))

        return self.embedding



