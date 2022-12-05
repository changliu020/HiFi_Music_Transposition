import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np

from dataset import NottinghamDataset
from torch.nn.utils import rnn
import random


class log_gaussian:

    def __call__(self, x, mu, var):

        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)

        return logli.sum(1).mean().mul(-1)


class Trainer:

    def __init__(self, G, FE, D, Q, device):

        self.G = G.to(device)
        self.FE = FE.to(device)
        self.D = D.to(device)
        self.Q = Q.to(device)

        self.device = device

        self.batch_size = 10

    def _noise_sample(self, dis_c, con_c, noise, bs):

        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs), idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        # noise.data.uniform_(-1.0, 1.0)
        dis_c = dis_c.unsqueeze(1).repeat(1, noise.shape[1], 1)
        con_c = con_c.unsqueeze(1).repeat(1, noise.shape[1], 1)
        z = torch.cat([noise, dis_c, con_c], 2)
        # z = z.view(-1, 74, 1, 1)

        return z, idx

    def train(self):

        # real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).to(self.device)
        label = torch.FloatTensor(self.batch_size, 1).to(self.device)
        dis_c = torch.FloatTensor(self.batch_size, 10).to(self.device)
        con_c = torch.FloatTensor(self.batch_size, 2).to(self.device)
        # noise = torch.FloatTensor(self.batch_size, 62).to(self.device)

        # real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        # noise = Variable(noise)

        criterionD = nn.BCELoss().to(self.device)
        criterionQ_dis = nn.CrossEntropyLoss().to(self.device)
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params': self.FE.parameters()}, {
                            'params': self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params': self.G.parameters()}, {
                            'params': self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

        # dataset = dset.MNIST(
        #     './dataset', transform=transforms.ToTensor(), download=True)
        # dataloader = DataLoader(
        #     dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        def collate_fn(data):
            data.sort(key=lambda d: d.shape[0], reverse=True)
            data_length = [d.shape[0] for d in data]
            data = rnn.pad_sequence(data, batch_first=True, padding_value=0)
            return data, data_length

        dataset = NottinghamDataset(
            './data/nottingham-dataset/wav', 22050, transformation='mfcc')
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=collate_fn)

        # fixed random variables
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 1, 0).reshape(-1, 1)

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])

        idx = np.arange(10).repeat(1)
        one_hot = np.zeros((self.batch_size, 10))
        one_hot[range(self.batch_size), idx] = 1
        # fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)

        num_epoch = 5
        for epoch in range(num_epoch):
            if epoch == 0:
                loss_arr = {'Iter': [], 'D_loss': [], 'G_loss': []}
            for num_iters, batch_data in enumerate(dataloader, 0):

                # real part
                optimD.zero_grad()

                x, x_len = batch_data
                x = x.to(self.device)

                bs = x.size(0)
                # real_x.data.resize_(x.size())
                label.data.resize_(bs, 1)
                # dis_c.data.resize_(bs, 10)
                # con_c.data.resize_(bs, 2)
                # noise.data.resize_(bs, 62)

                # real_x.data.copy_(x)
                x = rnn.pack_padded_sequence(x, x_len, batch_first=True)

                fe_out1 = self.FE(x)
                fe_out1, fe_out_len = rnn.pad_packed_sequence(fe_out1, batch_first=True)
                # zeros_c = torch.zeros_like(con_c)
                # fe_out1 = torch.cat((fe_out1, zeros_c.unsqueeze(1).repeat(1, fe_out1.shape[1], 1)), 2)

                probs_real = self.D(fe_out1)
                label.data.fill_(1)

                # label = torch.ones((bs, 1)).to(self.device)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z, idx = self._noise_sample(dis_c, con_c, fe_out1, bs)

                # fe_out1_cat = torch.cat((con_c.unsqueeze(1).repeat(1, fe_out1.shape[1], 1), fe_out1), 2)
                z = rnn.pack_padded_sequence(z, fe_out_len, batch_first=True)

                fake_x = self.G(z)
                fake_x, fake_x_len = rnn.pad_packed_sequence(fake_x, batch_first=True)
                fake_x = rnn.pack_padded_sequence(fake_x.detach(), fake_x_len, batch_first=True)

                fe_out2 = self.FE(fake_x)
                fe_out2, _ = rnn.pad_packed_sequence(fe_out2, batch_first=True)
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                fe_out, _ = rnn.pad_packed_sequence(fe_out, batch_first=True)

                probs_fake = self.D(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = self.Q(fe_out)
                class_ = torch.LongTensor(idx).to(self.device)
                target = Variable(class_)
                dis_loss = criterionQ_dis(q_logits, target)
                con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1

                G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()

                if epoch == 0:
                    loss_arr['Iter'].append(num_iters)
                    loss_arr['D_loss'].append(D_loss.data.cpu().numpy())
                    loss_arr['G_loss'].append(G_loss.data.cpu().numpy())

                if num_iters % 10 == 0:

                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )

                    # noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))

                    con_c.data.copy_(torch.from_numpy(c1))
                    z, _ = self._noise_sample(dis_c, con_c, fe_out1, bs)
                    z = rnn.pack_padded_sequence(z, fe_out_len, batch_first=True)

                    x_save = self.G(z)
                    x_save, _ = rnn.pad_packed_sequence(x_save, batch_first=True)
                    # save_image(x_save[0, :, :], './tmp/c1.png', nrow=10)
                    torch.save(x_save[0, :, :], './tmp/c1.pt')

                    # z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
                    # x_save = self.G(z)
                    # save_image(x_save.data, './tmp/c2.png', nrow=10)
                    con_c.data.copy_(torch.from_numpy(c2))
                    z, _ = self._noise_sample(dis_c, con_c, fe_out1, bs)
                    z = rnn.pack_padded_sequence(z, fe_out_len, batch_first=True)

                    x_save = self.G(z)
                    x_save, _ = rnn.pad_packed_sequence(x_save, batch_first=True)
                    # save_image(x_save[0, :, :], './tmp/c1.png', nrow=10)
                    torch.save(x_save[0, :, :], './tmp/c2.pt')

            if epoch == 0:
                torch.save(loss_arr, './tmp/loss_arr.pt')
            
        torch.save({'epoch_cnt': num_epoch, 'trainer': self},
                    './checkpoints' + '/' + self.generate_random_str(5) + '.pt')


    def generate_random_str(self, randomLength=8):
        random_str = ''
        base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
        length = len(base_str) - 1
        for _ in range(randomLength):
            random_str += base_str[random.randint(0, length)]
        return random_str