import torch
from torch.autograd import Variable
from nn import BuildModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from scalers import get_scaler


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
        self.gen_scheduler = CosineAnnealingLR(optimizer=self.optimizer_g, T_max=config['num_epochs'])
        self.disc_scheduler = CosineAnnealingLR(optimizer=self.optimizer_d, T_max=config['num_epochs'])
        self.scaler = get_scaler(scaler_type=config['scaler'])
        self.data_version = config['data_version']

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def make_fake(self, features):
        noise = torch.normal(0, 1, size=(features.shape[0], self.latent_dim), device=self.device)
        return self.generator(torch.cat((features, noise), axis=-1)).view(-1, 1, 8, 16)

    def gen_step(self, features):
        self.generator.zero_grad()
        errG = self.gen_loss(features)
        errG.backward()
        self.optimizer_g.step()
        return errG

    def gen_loss(self, features, requires_grad=True):
        fake_images = self.make_fake(features)
        fake_output = self.discriminator([features, fake_images])
        errG = -torch.mean(fake_output)
        return errG

    def disc_step(self, batch, features):
        self.discriminator.zero_grad()
        errD = self.disc_loss(batch, features)
        errD.backward()
        self.optimizer_d.step()
        return errD

    def disc_loss(self, batch, features, requires_grad=True):
        real_output = self.discriminator([features, batch])
        errD_real = torch.mean(real_output)
        fake_images = self.make_fake(features)
        fake_output = self.discriminator([features, fake_images])
        errD_fake = torch.mean(fake_output)
        gradient_penalty = self.calculate_penalty(self.discriminator,
                                                  batch.data, fake_images.data,
                                                  features,
                                                  self.device)
        errD = -errD_real + errD_fake + gradient_penalty * self.gp_lambda
        return errD

    def scheduler_step(self):
        self.gen_scheduler.step()
        self.disc_scheduler.step()

    def calculate_penalty(self, model, real, fake, features, device):
        alpha = torch.randn((real.size(0), 1, 1, 1), device=device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)

        model_interpolates = model([features, interpolates])
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

