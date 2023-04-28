import torch
import numpy as np
from tqdm import trange
import wandb
import random
from torch.autograd import Variable
import os

from callbacks import ScheduleLRCallback, WriteHistSummaryCallback, SaveModelCallback

seed = 42  # for result reproducibility
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
##################### add debugging using asserts in the future versions


class Trainer(object):
    def __init__(self,
                 model,
                 train_loader,
                 val_sample,
                 epochs,
                 batch_size,
                 num_disc_updates,
                 save_period,
                 noise_power, #add support for feature noise
                 noise_decay
                 ):

        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_disc_updates = num_disc_updates
        self.train_loader = train_loader
        self.Y_val, self.X_val = val_sample
        self.save_period = save_period
        self.noise_power = noise_power
        self.noise_decay = noise_decay
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train_step(self):
        self.model.train()
        loss_history = {'disc_losses': [], 'gen_losses': [], 'reg_losses': []}

        for epoch, (data, features) in enumerate(self.train_loader):
            real_images = data.to(self.device)
            features = features.type(torch.float32).clone().detach().requires_grad_(True).to(self.device)
            disc_loss, reg_loss = self.model.disc_step(real_images, features)

            if epoch % self.num_disc_updates == 0:
                gen_loss = self.model.gen_step(real_images, features)
                loss_history['reg_losses'].append(reg_loss)
                loss_history['disc_losses'].append(disc_loss)
                loss_history['gen_losses'].append(gen_loss)

        disc_loss = torch.mean(torch.tensor(loss_history['disc_losses']))
        gen_loss = torch.mean(torch.tensor(loss_history['gen_losses']))
        reg_loss = torch.mean(torch.tensor(loss_history['reg_losses']))
        return disc_loss, gen_loss, reg_loss

    def test_step(self):
        self.model.eval()
        losses = np.array([0, 0, 0], dtype='float32')
        for epoch in range(0, len(self.X_val), self.batch_size):
            features = torch.tensor(self.X_val[epoch:epoch + self.batch_size],
                                    requires_grad=False,
                                    device=self.device,
                                    dtype=torch.float32)
            data = torch.tensor(self.Y_val[epoch:epoch + self.batch_size],
                                requires_grad=False,
                                device=self.device,
                                dtype=torch.float32)
            g_loss = self.model.gen_loss(data, features)
            d_loss, r_loss = self.model.disc_loss(data, features)
            losses += np.array([d_loss.item(), g_loss.item(), r_loss.item()])
        losses /= (len(self.X_val) / self.batch_size)
        return losses[0], losses[1], losses[2]

    def train(self):
        wandb.login(key='b5bb9b937300c5d613b3a95f676708e5a88d2b7e')     # remove during commit
        wandb.init(project="coursework", entity="karimdzan")


        summary_callback = WriteHistSummaryCallback(model=self.model,
                                                    sample=(self.X_val, self.Y_val),
                                                    save_period=self.save_period)
        schedulelr = ScheduleLRCallback(self.model)

        if not os.path.isdir('checkpoints'):
            os.mkdir(
                'checkpoints'
            )

        saveModel = SaveModelCallback(model=self.model,
                                      path='checkpoints',
                                      save_period=self.save_period
                                      )
        callbacks = [summary_callback, schedulelr, saveModel]

        for epoch in trange(self.epochs):
            disc_loss, gen_loss, reg_loss_train = self.train_step()
            val_loss_disc, val_loss_gen, reg_loss_val = self.test_step()
            wandb.log({
                "Epoch": epoch,
                "gen loss train": gen_loss,
                "disc loss train": disc_loss,
                "reg_loss_train": reg_loss_train,
                "reg_loss_val": reg_loss_val,
                "gen loss val": val_loss_gen,
                "disc loss val": val_loss_disc
            })
            for f in callbacks:
                f(epoch)

        wandb.finish()

    def features_noise(self, epoch):
        current_power = self.noise_power / (10 ** (epoch / self.noise_decay))
        wandb.log({f"features noise power at epoch-epoch": current_power})
        return current_power



