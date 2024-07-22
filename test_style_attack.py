import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torchvision.utils as vutils
from facenet_pytorch import InceptionResnetV1
from dataset import get_MyData
from utils import Batch_Normalization, l2_norm, fix_randon_seed

from mask_face.mask_functions import tensor2Image
from skimage.metrics import structural_similarity as compare_ssim
from config import cfg_common, cfg_train_gpu0, cfg_other_targets
import warnings

warnings.filterwarnings('ignore')


def test_style_attack(cfg):
    seed = cfg['seed']
    fix_randon_seed(seed)

    # checkpoint path
    style_model_path = cfg['style_model_path']

    # data setting
    data_path = cfg['data_path']
    test_set = cfg['test_set']
    target_identity = cfg['target_identity']
    target_img = cfg['target_img']
    img_size = cfg['img_size']

    # training parameters
    model_lr = cfg['model_lr']
    dist_lambda = cfg['dist_lambda']
    l1_lambda = cfg['l1_lambda']

    # face model setting
    face_model_path = cfg['face_model_path']
    backbone_name = cfg['backbone_name']
    head_name = cfg['head_name']
    dist_threshold = cfg['dist_threshold']
    img_mean = cfg['img_mean']
    img_std = cfg['img_std']

    # gpu setting
    device = cfg['device']
    pin_memory = cfg['pin_memory']
    num_workers = cfg['num_workers']

    # other parameters
    temperature = cfg['temperature']

    # load face model
    checkpoint_road = f'{face_model_path}/{backbone_name}_{head_name}.pth'
    checkpoint = torch.load(checkpoint_road)
    face_model = checkpoint['backbone'].eval().to(device)

    # load style attack model
    path = f'./checkpoint/attack_models/other_targets/{backbone_name}_{target_identity}_' \
           f'best_d{str(int(dist_lambda))}_t{temperature}_l1{l1_lambda}.t7'

    attack_checkpoint = torch.load(path)
    print(path)
    attack_model = attack_checkpoint['model'].eval().to(device)
    weights = attack_checkpoint['weights'].detach().to(device)
    print(weights)

    # load original style model
    original_checkpoint = torch.load(style_model_path)
    original_style_model = original_checkpoint['model']
    original_style_model = original_style_model.eval().to(device)

    # BN
    face_bn = Batch_Normalization(mean=img_mean, std=img_std, device=device)

    # data
    batch_size = 1
    my_loader = get_MyData(batch_size=batch_size, mode=test_set, pin_memory=pin_memory,
                           num_workers=num_workers, target_identity=target_identity, shuffle=False)

    # target face & target embedding
    target_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    target_path = f'{data_path}/{target_identity}/{target_img}'
    target_face_img = target_transform(Image.open(target_path)).unsqueeze(0).to(device)
    target_embedding_ori = l2_norm(face_model(face_bn.norm(target_face_img)))

    save_img_path = f'./results/{target_identity}_style_attack'
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)


    total, attack_same_count, total_ssim = 0, 0, 0.
    i = 0
    for idx, [face_imgs, mask_bins, mask_imgs] in enumerate(my_loader):
        total += face_imgs.size(0)
        target_embedding = target_embedding_ori.repeat(face_imgs.size(0), 1)
        face_imgs, mask_bins, mask_imgs = face_imgs.to(device), mask_bins.to(device), mask_imgs.to(device)

        batch_weights = weights.unsqueeze(0).repeat(face_imgs.size(0), 1)
        batch_weights = torch.softmax(batch_weights / temperature, dim=1)

        style_ori_mask = original_style_model(mask_imgs, batch_weights, True)
        style_ori_face = face_imgs * (1 - mask_bins) + style_ori_mask * mask_bins
        style_attack_mask = attack_model(mask_imgs, batch_weights, True)
        style_attack_face = face_imgs * (1 - mask_bins) + style_attack_mask * mask_bins

        attack_embedding = l2_norm(face_model(face_bn.norm(style_attack_face)))
        attack_diff = torch.subtract(attack_embedding, target_embedding)
        attack_dist = torch.sum(torch.square(attack_diff), dim=1).item()

        attack_same_count += 1 if attack_dist < dist_threshold else 0

        total_ssim += compare_ssim(np.array(tensor2Image(style_attack_face.squeeze(0))),
                                   np.array(tensor2Image(style_ori_face.squeeze(0))),
                                   win_size=111, data_range=255, multichannel=True)
        vutils.save_image(style_attack_face, f'{save_img_path}/{i}_style_attack_face.jpg')
        vutils.save_image(style_ori_face, f'{save_img_path}/{i}_style_ori_face.jpg')
        vutils.save_image(face_imgs, f'{save_img_path}/{i}_ori_face.jpg')
        i += 1

    ASR = attack_same_count / total
    SSIM = total_ssim / total

    print(f'Testing of {target_identity}, attack ASR: {ASR:.4f}, ssim: {SSIM:.4f}')


if __name__ == '__main__':
    targets = ['Steve_Park', 'Oliver_Phelps', 'Tommy_Shane_Steiner', 'Michael_Chang']
    for target in targets:
        config = {**cfg_common, **cfg_train_gpu0, **cfg_other_targets[target]}
        test_style_attack(config)
