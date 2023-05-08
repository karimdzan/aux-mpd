import torch
from torch.autograd import Variable
from nn import BuildModel
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, CosineAnnealingWarmRestarts
from scalers import get_scaler
import torch.nn as nn
from metrics.gaussian_metrics import get_val_metric_v


class Model(object):
    def __init__(self, config):
        if config['data_version'] == 'data_v4plus':
            self.full_feature_space = config.get('full_feature_space', False)
            self.include_pT_for_evaluation = config.get('include_pT_for_evaluation', False)
        self.step_counter = 0
        self.NUM_DISC_UPDATES = config['num_disc_updates']
        architecture = config['architecture']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.generator = BuildModel(architecture['generator']).to(self.device)
        self.discriminator = BuildModel(architecture['discriminator']).to(self.device)
        self.optimizer_g = torch.optim.RMSprop(self.generator.parameters(), lr=config['lr_gen'])
        self.optimizer_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=config['lr_disc'])
        self.gp_lambda = config['gp_lambda']
        self.latent_dim = config['latent_dim']
        self.batch_size = config['batch_size']
        self.gen_scheduler = CosineAnnealingWarmRestarts(optimizer=self.optimizer_g, T_0=config['num_epochs'])
        self.disc_scheduler = CosineAnnealingWarmRestarts(optimizer=self.optimizer_d, T_0=config['num_epochs'])
        self.scaler = get_scaler(scaler_type=config['scaler'])
        self.data_version = config['data_version']
        self.LOSS_WEIGHT = config['loss_weight']

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def make_fake(self, features):
        noise = torch.normal(0, 1, size=(features.shape[0], self.latent_dim), device=self.device)
        return self.generator(torch.cat((features, noise), axis=-1))[0].view(-1, 1, 8, 16)

    def gen_step(self, batch, features):
        self.generator.zero_grad()
        errG = self.gen_loss(batch, features)
        errG.backward()
        self.optimizer_g.step()
        return errG

    def gen_loss(self, batch, features):
        real_output = self.discriminator([features, batch])[0]
        errD_real = torch.mean(real_output)
        fake_images = self.make_fake(features)
        fake_output = self.discriminator([features, fake_images])[0]
        errD_fake = torch.mean(fake_output)
        return errD_real - errD_fake

    def disc_step(self, batch, features):
        self.discriminator.zero_grad()
        errD, reg_loss = self.disc_loss(batch, features)
        errD.backward()
        self.optimizer_d.step()
        return errD, reg_loss.detach().clone()

    def disc_loss(self, batch, features):
        output_real  = self.discriminator([features, batch])
        real_output, reg_real = output_real[0], output_real[1]
        errD_real = torch.mean(real_output)

        fake_images = self.make_fake(features)
        output_fake = self.discriminator([features, fake_images])
        fake_output, reg_fake = output_fake[0], output_fake[1]
        real_metric = get_val_metric_v(self.scaler.unscale(batch.view(-1, 8, 16).cpu().detach().numpy())).T[-3]
        reg_loss = nn.MSELoss()(reg_fake,
                                torch.tensor(real_metric, requires_grad=True, device=self.device, dtype=torch.float32).view(-1, 1))
        errD_fake = torch.mean(fake_output)

        if self.generator.training and self.discriminator.training:
            gradient_penalty = self.calculate_penalty(self.discriminator,
                                                      batch.data, fake_images.data,
                                                      features,
                                                      self.device)
            errD = self.LOSS_WEIGHT * (-errD_real + errD_fake + gradient_penalty * self.gp_lambda) + reg_loss
        else:
            errD = self.LOSS_WEIGHT * (-errD_real + errD_fake) + reg_loss
        return errD, reg_loss

    def calculate_penalty(self, model, real, fake, features, device):
        alpha = torch.randn((real.size(0), 1, 1, 1), device=device)
        interpolates = (alpha * real.view(-1, 1, 8, 16) + ((1 - alpha) * fake)).requires_grad_(True)

        model_interpolates = model([features, interpolates])[0]
        grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

