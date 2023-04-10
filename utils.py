from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import logging
import random
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
import argparse
from sklearn.model_selection import train_test_split
from data.preprocessing import read_csv_2d


logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def make_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--checkpoint_name', type=str, required=False)
    parser.add_argument('--gpu_num', type=str, required=False)
    parser.add_argument('--prediction_only', action='store_true', default=False)
    parser.add_argument('--logging_dir', type=str, default='logs')

    return parser


def print_args(args):
    print()
    print("----" * 10)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"    {k} : {v}")
    print("----" * 10)
    print()


def parse_args():
    args = make_parser().parse_args()
    print_args(args)
    return args


def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert (config['feature_noise_power'] is None) == (
            config['feature_noise_decay'] is None
    ), 'Noise power and decay must be both provided'

    if 'lr_disc' not in config:
        config['lr_disc'] = config['lr']
    if 'lr_gen' not in config:
        config['lr_gen'] = config['lr']

    return config


def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a
    Args:
        seed (int): The desired seed.
    """

    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    logger.info("Initialize random seed.")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_features(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    #   padrow {23, 33}
    #   pT [0, 2.5]
    # if not torch.is_tensor(features):
    #     features = torch.tensor(features)
    bin_fractions = features[:, 2:4].cpu() % 1
    features_1 = (features[:, :3].cpu() - torch.tensor([[0.0, 0.0, 162.5]])) / torch.tensor([[20.0, 60.0, 127.5]])
    features_2 = (features[:, 4:5] >= 27).type(torch.DoubleTensor)
    features_3 = features[:, 5:6] / 2.5
    return torch.cat((features_1, features_2, features_3, bin_fractions), dim=-1)


class LoadData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, features = self.dataset[idx]
        return self.transform(data), self.transform(features)


def load_data(data_version, scaler, transform, bs):
    data, features = read_csv_2d(filename='data/' + data_version + '/csv/digits.csv', strict=False)
    data = scaler.scale(data)
    data = np.float32(data)
    features = np.float32(features)
    Y_train, Y_test, X_train, X_test = train_test_split(data, features, test_size=0.25, random_state=42)
    train_dataset = LoadData([Y_train, X_train], transform)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=4, pin_memory=True)
    return train_loader, (Y_test, X_test)
