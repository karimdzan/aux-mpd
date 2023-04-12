import torch
import numpy as np
from tqdm import tqdm
import wandb
import random
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

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
        self.writer_train = SummaryWriter()
        self.writer_val = SummaryWriter()
        self.noise_power = noise_power
        self.noise_decay = noise_decay
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train_step(self):
        self.model.train()
        loss_history = {'disc_losses': [], 'gen_losses': []}
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for epoch, (data, features) in progress_bar:
            real_images = data.to(self.device)
            features = torch.tensor(features, requires_grad=True).to(self.device)

            disc_loss = self.model.disc_step(real_images, features)

            if epoch % self.num_disc_updates == 0:
                gen_loss = self.model.gen_step(features)

                loss_history['disc_losses'].append(disc_loss)
                loss_history['gen_losses'].append(gen_loss)

        disc_loss = torch.mean(torch.tensor(loss_history['disc_losses']))
        gen_loss = torch.mean(torch.tensor(loss_history['gen_losses']))
        return disc_loss, gen_loss

    def test_step(self):
        self.model.eval()
        losses = np.array([0, 0], dtype='float32')
        for epoch in range(0, len(self.X_val), self.batch_size):
            features = Variable(torch.from_numpy(self.X_val[epoch:epoch + self.batch_size]),
                                requires_grad=False).to(self.device)
            data = Variable(torch.from_numpy(self.Y_val[epoch:epoch + self.batch_size]),
                            requires_grad=False).to(self.device)
            g_loss = self.model.gen_loss(features)
            d_loss = self.model.disc_loss(data, features)
            losses += np.array([d_loss.item(), g_loss.item()])
        losses /= (len(self.X_val) / self.batch_size)
        return losses[0], losses[1]

    def train(self):
        wandb.login(key='b5bb9b937300c5d613b3a95f676708e5a88d2b7e')     # remove during commit
        wandb.init(project="coursework", entity="karimdzan")

        loss_history = {'disc_losses': [], 'gen_losses': []}

        summary_callback = WriteHistSummaryCallback(model=self.model,
                                                    sample=(self.X_val, self.Y_val),
                                                    save_period=self.save_period,
                                                    writer=self.writer_val
                                                    )
        schedulelr = ScheduleLRCallback(self.model,
                                        self.writer_val
                                        )

        saveModel = SaveModelCallback(model=self.model,
                                      path='checkpoints',
                                      save_period=self.save_period
                                      )
        callbacks = [summary_callback, schedulelr, saveModel]

        for epoch in tqdm(range(self.epochs)):
            disc_loss, gen_loss = self.train_step()
            val_loss_disc, val_loss_gen = self.test_step()
            wandb.log({
                "Epoch": epoch,
                "gen loss train": gen_loss,
                "disc loss train": disc_loss,
                "gen loss val": val_loss_gen,
                "disc loss val": val_loss_disc
            })

            for f in callbacks:
                f(epoch)

        wandb.finish()

    def features_noise(self, epoch):
        current_power = self.noise_power / (10 ** (epoch / self.noise_decay))
        self.writer_train.add_scalar("features noise power", current_power, epoch)
        return current_power



