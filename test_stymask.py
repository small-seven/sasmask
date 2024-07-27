import os
import torch
import warnings
import numpy as np
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

from config import *
from dataset import get_MyData
from utils import Batch_Normalization, fix_randon_seed, l2_norm
from mask_face.mask_functions import tensor2Image

from utils import get_landmark_detector

from nn_modules import LandmarkExtractor, FaceXZooProjector

warnings.filterwarnings('ignore')


def test_stymask(cfg, target_model='ResNet_50_ArcFace'):
    seed = cfg['seed']
    fix_randon_seed(seed)

    style_model_path = cfg['style_model_path']

    # data setting
    data_path = cfg['data_path']
    test_set = cfg['test_set']
    target_identity = cfg['target_identity']
    target_img = cfg['target_img']
    img_size = cfg['img_size']

    # face model setting
    face_model_path = cfg['face_model_path']
    backbone_name = cfg['backbone_name']
    head_name = cfg['head_name']
    img_mean = cfg['img_mean']
    img_std = cfg['img_std']

    dist_threshold = cfg['dist_threshold']

    # gpu setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = cfg['pin_memory']

    # load face model
    model_path = f'{face_model_path}/{target_model}.pth'
    checkpoint = torch.load(model_path)
    face_model = checkpoint['backbone']
    face_model = face_model.eval().to(device)

    # BN
    face_bn = Batch_Normalization(mean=img_mean, std=img_std, device=device)

    # data
    batch_size = 1
    my_loader = get_MyData(batch_size=batch_size, mode=test_set, pin_memory=pin_memory,
                           num_workers=4, target_identity=target_identity, shuffle=False)

    # target face & target embedding
    target_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    target_path = f'{data_path}/{target_identity}/{target_img}'
    target_face_img = target_transform(Image.open(target_path)).unsqueeze(0).to(device)
    target_embedding_ori = l2_norm(face_model(face_bn.norm(target_face_img)))

    face_landmark_detector = get_landmark_detector(
        landmark_detector_type='mobilefacenet', device=device
    )
    location_extractor = LandmarkExtractor(device, face_landmark_detector, (112, 112)).to(device)
    fxz_projector = FaceXZooProjector(device=device, img_size=(112, 112), patch_size=(112, 112)).to(device)

    # advmask
    uv_mask = transforms.ToTensor()(Image.open('./prnet/new_uv.png').convert('L')).unsqueeze(0).to(device)
    original_pattern = target_transform(Image.open('./mask_face/masks/textures/23.jpg').convert('RGB')).unsqueeze(0).to(
        device)

    load_path = f"./checkpoint/{cfg['target_identity']}_attack_model_epoch_0.pth"
    attack_checkpoint = torch.load(load_path)
    attack_model = attack_checkpoint['model'].to(device)
    weights = attack_checkpoint['weights']
    temperature = attack_checkpoint['temperature']

    original_checkpoint = torch.load(cfg['style_model_path'])
    original_model = original_checkpoint['model'].to(device)

    # stymask_path = f'./data/trained_mask/saved_stymask/{target_identity}_{backbone_name}_{head_name}.png'
    # stymask = transforms.ToTensor()(Image.open(stymask_path).convert('RGB'))
    # stymask = stymask * uv_mask

    # orimask_path = f'./data/ori_styles/{style_selection}.jpg'
    # orimask = transforms.ToTensor()(Image.open(orimask_path).convert('RGB'))
    # orimask = orimask * uv_mask

    total, attack_same_count, ori_same_count, total_ssim = 0, 0, 0, 0.
    for idx, [face_imgs, _, _] in enumerate(my_loader):
        batch_weights = weights.unsqueeze(0).repeat(face_imgs.size(0), 1).to(device)
        batch_weights = torch.softmax(batch_weights / temperature, dim=1)
        with torch.no_grad():
            stymask = attack_model(original_pattern, batch_weights, True) * uv_mask
            orimask = original_model(original_pattern, batch_weights, True) * uv_mask
        face_imgs, stymask, orimask = face_imgs.to(device), stymask.to(device), orimask.to(device)

        preds = location_extractor(face_imgs)
        ori_faces = fxz_projector(face_imgs, preds, orimask, do_aug=True)
        sty_faces = fxz_projector(face_imgs, preds, stymask, do_aug=True)

        total += face_imgs.size(0)
        target_embedding = target_embedding_ori.repeat(face_imgs.size(0), 1)

        ori_embedding = l2_norm(face_model(face_bn.norm(ori_faces)))
        diff = torch.subtract(ori_embedding, target_embedding)
        ori_dist = torch.sum(torch.square(diff), dim=1)
        ori_same_count += 1 if ori_dist < dist_threshold else 0

        attack_embedding = l2_norm(face_model(face_bn.norm(sty_faces)))
        diff = torch.subtract(attack_embedding, target_embedding)
        attack_dist = torch.sum(torch.square(diff), dim=1)
        attack_same_count += 1 if attack_dist < dist_threshold else 0

        total_ssim += compare_ssim(np.array(tensor2Image(ori_faces.squeeze(0))),
                                   np.array(tensor2Image(sty_faces.squeeze(0))),
                                   win_size=111, data_range=255, multichannel=True)
    print(
        f'{target_model}\n'
        f'ori ASR: {ori_same_count / total:.4f}, attack ASR: {attack_same_count / total:.4f}\n'
        f'SSIM: {total_ssim / total:.4f}\n'
    )


if __name__ == "__main__":
    test_stymask(cfg, target_model='ResNet_50_ArcFace')
