import argparse
import warnings

import torch
from torchvision import transforms
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from config import *
from dataset import get_MyData
from utils import Batch_Normalization, fix_randon_seed, l2_norm, get_hms
from skimage.metrics import structural_similarity as compare_ssim
import utils2
import time
from nn_modules import LandmarkExtractor, FaceXZooProjector
from choose_data import get_identities, choose_data
from mask_face.mask_functions import tensor2Image
from torch.autograd import Variable

warnings.filterwarnings('ignore')


def train_pgdmask(cfg):
    # data
    train_loader = get_MyData(batch_size=cfg["batch_size"], mode=cfg["train_set"], pin_memory=False,
                              num_workers=8, target_identity=cfg["target_identity"], shuffle=False)
    test_loader = get_MyData(batch_size=cfg["batch_size"], mode=cfg["test_set"], pin_memory=False,
                             num_workers=8, target_identity=cfg["target_identity"], shuffle=False)
    return train_loader, test_loader


def parseArgs():
    parser = argparse.ArgumentParser(description="Training Adv StyMask.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=128)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    fix_randon_seed(args.seed)

    identities = get_identities()

    for chosen_identity in identities:
        print(chosen_identity)
        choose_data(chosen_identity)
        cfg_target = dict(
            target_identity=chosen_identity,
            target_img=f'{chosen_identity}_0001.jpg',
        )
        cfg_eps = dict(
            eps=args.eps / 255.,
            steps=40,
        )
        config = {
            **cfg_common, **cfg_devices[f'gpu{args.device}'],
            **cfg_sty_ab[1], **cfg_eps, **cfg_target,
        }
        train_pgdmask(config)
