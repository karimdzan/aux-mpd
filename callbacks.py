import torch
from metrics import make_images_for_model
import wandb
import os


class SaveModelCallback:
    def __init__(self, model, path, save_period):
        self.model = model
        self.optimizer_g = model.optimizer_g
        self.optimizer_d = model.optimizer_d
        self.path = path
        self.save_period = save_period

    def __call__(self, step):
        if step % self.save_period == 0:
            torch.save({'epoch': step,
                        'model_gen_state_dict': self.model.generator.state_dict(),
                        'model_disc_state_dict': self.model.discriminator.state_dict(),
                        'optimizer_gen_state_dict': self.optimizer_g.state_dict(),
                        'optimizer_disc_state_dict': self.optimizer_d.state_dict()},
                       f'{self.path}/model_{step}.pth')


class WriteHistSummaryCallback:
    def __init__(self, model, sample, save_period):
        self.model = model
        self.save_period = save_period
        self.sample = sample
        self.table = wandb.Table(columns=['ID', 'Image'])

    def __call__(self, step):
        if step % self.save_period == 0:
            images, images1, img_amplitude, chi2 = make_images_for_model(self.model, sample=self.sample, calc_chi2=True)
            wandb.log({'chi2': chi2})
            for k, img in images.items():
                img_log = wandb.Image(img)
                wandb.log({"images": img_log})
            for k, img in images1.items():
                img_log = wandb.Image(img)
                wandb.log({"Masked images": img_log})
            img_log = wandb.Image(img_amplitude)
            wandb.log({"images with amplitude": img_log})


class ScheduleLRCallback:
    def __init__(self, model):
        self.model = model
        self.sch_gen = model.gen_scheduler
        self.sch_disc = model.disc_scheduler

    def __call__(self, step):
        self.sch_gen.step()
        self.sch_disc.step()
        wandb.log({"discriminator learning rate at step": self.sch_gen.get_last_lr()[0]})
        wandb.log({"generator learning rate at step": self.sch_disc.get_last_lr()[0]})
