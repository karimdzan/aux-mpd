import torch
from metrics import make_images_for_model
import wandb


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
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_gen_state_dict': self.optimizer_g.state_dict(),
                        'optimizer_disc_state_dict': self.optimizer_d.state_dict()},
                       f'{self.path}/model_{step}.pth')


class WriteHistSummaryCallback:
    def __init__(self, model, sample, save_period, writer):
        self.model = model
        self.save_period = save_period
        self.writer = writer
        self.sample = sample

    def __call__(self, step):
        if step % self.save_period == 0:
            images, images1, img_amplitude, chi2 = make_images_for_model(self.model, sample=self.sample, calc_chi2=True)
            wandb.log({'chi2': chi2})
            self.writer.add_scalar("chi2", chi2, step)
            for k, img in images.item():
                self.writer.add_image(f"img{k}", img, step)
            for k, img in images1.item():
                self.writer.add_image(f"img{k} (amp > 1)", img, step)

            self.writer.add_image(f"log10(amplitude + 1)", img_amplitude, step)


class ScheduleLRCallback:
    def __init__(self, model, writer):
        self.model = model
        self.sch_gen = model.gen_scheduler
        self.sch_disc = model.disc_scheduler
        self.writer = writer

    def __call__(self, step):
        self.sch_gen.step()
        self.sch_disc.step()
        self.writer.add_scalar("discriminator learning rate", self.model.disc_opt.lr, step)
        self.writer.add_scalar("generator learning rate", self.model.gen_opt.lr, step)


#%%
