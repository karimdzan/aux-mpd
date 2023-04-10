from train import Trainer
from utils import parse_args, load_config
import torchvision.transforms as T
from utils import load_data
from scalers import get_scaler
from model import Model


def main():
    args = parse_args()
    # config_path = str(model_path / 'config.yaml')
    config_path = 'model/baseline_8x16_GP50_full_fc.yaml'  # add support for checkpoints importing
    args.config = config_path
    config = load_config(args.config)

    scaler = get_scaler(scaler_type=config['scaler'])
    transform = T.Compose([
        T.ToTensor()
    ])

    model = Model(config)

    train_loader, val_sample = load_data(config['data_version'], scaler, transform, config['batch_size'])

    trainer = Trainer(model,
                      train_loader,
                      val_sample,
                      config['num_epochs'],
                      config['batch_size'],
                      config['num_disc_updates'],
                      config['save_every'],
                      config['feature_noise_power'],
                      config['feature_noise_decay'])  # add support for predictions only and customize evaluation process

    trainer.train()


if __name__ == '__main__':
    main()

# %%

# %%
