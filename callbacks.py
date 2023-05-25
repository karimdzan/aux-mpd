import torch
from metrics import make_images_for_model
import wandb
import os
import PIL
import numpy as np
import io
from metrics.gaussian_metrics import get_val_metric_v
from scalers import get_scaler
import matplotlib.pyplot as plt


class SaveModelCallback:
    def __init__(self, model, path, save_period):
        self.model = model
        self.optimizer_g = model.optimizer_g
        self.optimizer_d = model.optimizer_d
        self.path = path
        self.save_period = save_period

    def __call__(self, step):
        if step % self.save_period == 0 and step != 0:
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.scaler = get_scaler("logarithmic")


    def __call__(self, step):
        if step % self.save_period == 0 and step != 0:
            x = torch.tensor(self.sample[0],
                             requires_grad=False,
                             device=self.device,
                             dtype=torch.float32)
            y = torch.tensor(self.sample[1], 
                             requires_grad=False,
                             device=self.device,
                             dtype=torch.float32).view(-1, 1, 8, 16)
            output = self.model.discriminator([x, y])[1].cpu().detach().numpy()
            real_metric = get_val_metric_v(self.scaler.unscale(y.view(-1, 8, 16).cpu().detach().numpy())).T[-3]
            plot = self.get_distr(real_metric.squeeze(), output.squeeze())
            img_log = wandb.Image(plot)
            wandb.log({"2d hist" : img_log})
            
            images, images1, img_amplitude, chi2, chi2_feature = make_images_for_model(self.model, sample=self.sample, calc_chi2=True)
            wandb.log({'chi2': chi2,
                       'chi2_Sigma1^2': chi2_feature
                      })
            print(chi2)
            print(chi2_feature)
            for k, img in images.items():
                img_log = wandb.Image(img)
                wandb.log({"images": img_log})
            for k, img in images1.items():
                img_log = wandb.Image(img)
                wandb.log({"Masked images": img_log})
            img_log = wandb.Image(img_amplitude)
            wandb.log({"images with amplitude": img_log})


    
    def get_distr(self, real, fake):
        plt.hist2d(real, fake, bins=70)
        plt.title("2d histogram of the real and fake values of the sigma1 metric")
        plt.xlabel("real value")
        plt.ylabel("fake value")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.tight_layout()
        plt.close()
        buf.seek(0)
        img = PIL.Image.open(buf)
        return img
            


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
