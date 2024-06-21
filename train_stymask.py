import argparse
import os
import torch
import warnings
from torch import nn
import numpy as np
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from vgg import Vgg16
import torch.nn.functional as F

from config import cfg_common, cfg_targets, cfg_devices, cfg_sty_ab, cfg_heads
from dataset import get_MyData
from utils import Batch_Normalization, fix_randon_seed, l2_norm, gram_matrix
from mask_face.mask_functions import tensor2Image
from torch import optim
import utils2
from utils2 import EarlyStopping2
import time
from datetime import date

from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation

warnings.filterwarnings('ignore')


def train_stymask(cfg):
    fix_randon_seed(cfg['seed'])
    print(f"dist_lambda: {cfg['dist_lambda']}, l1_lambda: {cfg['l1_lambda']}, "
          f"tv_lambda: {cfg['tv_lambda']}, percep_lambda: {cfg['percep_lambda']}, "
          f"style_lambda: {cfg['style_lambda']}, ")
    # print("Overall Configurations:")
    # print(cfg)

    # save name
    save_folder = f'./checkpoint/{date.today()}'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # print_str = f'\nmodel_lr: {cfg["model_lr"]}, weight_lr: {cfg["weight_lr"]}, dist_lambda: {cfg["dist_lambda"]}, ' \
    #             f't: {cfg["temperature"]}, l1: {cfg["l1_lambda"]}, '
    # if 'loss_ablation' in cfg:
    #     print_str += f'loss ablation: {cfg["loss_ablation"]}\n'
    # elif 'style_selection' in cfg:
    #     print_str += f'style selection: {cfg["style_selection"]}\n'
    # else:
    #     print_str += '\n'
    #
    # print(f'{print_str}'
    #       f'backbone: {cfg["backbone_name"]}, head: {cfg["head_name"]}, dist_threshold: {cfg["dist_threshold"]}\n'
    #       f'target: {cfg["target_img"]}, train_set:{cfg["train_set"]}\n')

    # load style model
    attack_checkpoint = torch.load(cfg['style_model_path'])
    attack_model = attack_checkpoint['model'].to(cfg["device"])
    original_checkpoint = torch.load(cfg['style_model_path'])
    original_model = original_checkpoint['model'].eval().to(cfg["device"])

    # load face model
    model_path = f'{cfg["face_model_path"]}/{cfg["backbone_name"]}_{cfg["head_name"]}.pth'
    checkpoint = torch.load(model_path)
    face_model = checkpoint['backbone']
    face_model = face_model.eval().to(cfg["device"])
    vgg = Vgg16().to(cfg["device"]).eval()

    # BN
    face_bn = Batch_Normalization(mean=cfg["img_mean"], std=cfg["img_std"], device=cfg["device"])
    vgg_bn = Batch_Normalization(mean=cfg["vgg_mean"], std=cfg["vgg_std"], device=cfg["device"])

    # data
    batch_size = 1
    my_loader = get_MyData(batch_size=batch_size, mode=cfg["train_set"], pin_memory=cfg["pin_memory"],
                           num_workers=cfg["num_workers"], target_identity=cfg["target_identity"], shuffle=False)

    # optimizing setting
    optimizer_model = optim.Adam(attack_model.parameters(), lr=cfg["model_lr"], amsgrad=True)
    scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, patience=2, min_lr=1e-6, mode='min')
    weights = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).to(cfg["device"])
    if 'style_selection' in cfg:
        cfg["weight_lr"] = 0
        weights[cfg["style_selection"]] = 1
    else:
        weights.requires_grad = True
        optimizer_weight = optim.SGD([weights], lr=cfg["weight_lr"])
        # scheduler_weight = scheduler_factory(optimizer_weight)

    # target img
    target_transform = transforms.Compose([
        transforms.Resize(cfg["img_size"]),
        transforms.ToTensor(),
    ])
    target_path = f'{cfg["data_path"]}/{cfg["target_identity"]}/{cfg["target_img"]}'
    target_face_img = target_transform(Image.open(target_path)).unsqueeze(0).to(cfg["device"])
    with torch.no_grad():
        target_embedding_ori = l2_norm(face_model(face_bn.norm(target_face_img)))

    # style imgs setting
    transform_style = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    n_styles = 10
    style_imgs = torch.stack(
        [transform_style(Image.open(f'./data/style_images/{i}.jpg').convert('RGB')).unsqueeze(0).to(cfg["device"]) for
         i in range(n_styles)]
    )
    style_imgs_norm = vgg_bn.norm(style_imgs.clone())

    # face detector model
    face_landmark_detector = utils2.get_landmark_detector(
        landmark_detector_type='mobilefacenet', device=cfg["device"]
    )
    location_extractor = LandmarkExtractor(cfg["device"], face_landmark_detector, (112, 112)).to(cfg["device"])
    fxz_projector = FaceXZooProjector(device=cfg["device"], img_size=(112, 112), patch_size=(112, 112)).to(
        cfg["device"])

    # tv loss
    total_variation = TotalVariation(cfg["device"]).to(cfg["device"])
    l1_creterion = nn.L1Loss().to(cfg["device"])

    # optimizing setting
    uv_mask = transforms.ToTensor()(Image.open('./prnet/new_uv.png').convert('L')).unsqueeze(0).to(cfg["device"])
    original_pattern = target_transform(Image.open('./mask_face/masks/textures/23.jpg').convert('RGB')).unsqueeze(0)
    # style_attack_mask = original_pattern * uv_mask  # original adv stymask
    # early stop
    early_stop = EarlyStopping2(patience=7)

    # saved folder
    saved_fold = './results/saved_stymask/'
    if not os.path.exists(saved_fold):
        os.mkdir(saved_fold)
    saved_name = f'l1-{cfg["l1_lambda"]}_ltv-{cfg["tv_lambda"]}' \
                 f'_lc-{cfg["percep_lambda"]}_ls-{cfg["style_lambda"]}.png'
    saved_path = f'{saved_fold}/{saved_name}'

    epoch_length = len(my_loader)
    train_losses_epoch = []

    elapsed_time = 0

    for epoch in range(cfg["num_epoch"]):
        start_time = time.time()
        imgs_all, loss_tv_all, loss_l1_all, loss_dist_all, loss_percep_all, loss_style_all = 0, 0., 0., 0., 0., 0.
        train_loss = 0.0
        lr = optimizer_model.param_groups[0]['lr']
        # lr_weight = optimizer_weight.param_groups[0]['lr']
        attack_model.train()
        for idx, [face_imgs, mask_bins, mask_imgs] in enumerate(my_loader):
            with torch.no_grad():
                target_embedding = target_embedding_ori.repeat(face_imgs.size(0), 1)
                face_imgs, mask_bins, mask_imgs = face_imgs.to(cfg["device"]), mask_bins.to(
                    cfg["device"]), mask_imgs.to(cfg["device"])

            original_pattern = original_pattern.to(cfg["device"])

            batch_weights = weights.unsqueeze(0).repeat(face_imgs.size(0), 1).to(cfg["device"])
            if 'style_selection' not in cfg:
                batch_weights = torch.softmax(batch_weights / cfg["temperature"], dim=1)
            style_attack_pattern = attack_model(original_pattern, batch_weights, True)
            with torch.no_grad():
                real_style_pattern = original_model(original_pattern, batch_weights, True)

            style_attack_mask = style_attack_pattern * uv_mask

            preds = location_extractor(face_imgs)

            style_masked_face = fxz_projector(face_imgs, preds, style_attack_mask, do_aug=True)
            style_masked_face = torch.clamp(style_masked_face, min=0., max=1.)

            # First, TV loss
            loss_tv = total_variation(style_attack_mask) * cfg['tv_lambda']

            # Second, G(A) = B, L1 loss
            loss_l1 = l1_creterion(style_attack_pattern, real_style_pattern) * cfg["l1_lambda"]

            # Third, adv loss
            attack_embedding = l2_norm(face_model(face_bn.norm(style_masked_face)))
            diff = torch.subtract(attack_embedding, target_embedding)
            loss_dist = torch.sum(torch.square(diff), dim=1) * cfg["dist_lambda"]

            # Forth, style attack mask = real attack mask
            with torch.no_grad():
                real_style_feat = vgg(vgg_bn.norm(mask_imgs))
            stylized_feat = vgg(vgg_bn.norm(style_attack_mask))

            # Forth, content loss
            p1 = F.mse_loss(stylized_feat.relu1_2, real_style_feat.relu1_2)
            p2 = F.mse_loss(stylized_feat.relu2_2, real_style_feat.relu2_2)
            p3 = F.mse_loss(stylized_feat.relu3_3, real_style_feat.relu3_3)
            p4 = F.mse_loss(stylized_feat.relu4_3, real_style_feat.relu4_3)
            loss_perceptual = (p1 + p2 + p3 + p4) * cfg["percep_lambda"]

            # Fifth, style loss
            if torch.abs(weights).sum().item() != 0:
                weights_onehot = torch.zeros_like(weights)
                weights_onehot[weights.argmax()] = 1.
                batch_weights_onehot = weights_onehot.unsqueeze(0).repeat(face_imgs.size(0), 1).to(cfg["device"])
                style_imgs_inputs = torch.stack(
                    [style_imgs_norm[torch.nonzero(batch_weights_onehot)[i][1]].squeeze(0) for i in
                     range(face_imgs.size(0))], dim=0)

                with torch.no_grad():
                    style_feat = vgg(style_imgs_inputs)

                s1 = F.mse_loss(gram_matrix(stylized_feat.relu1_2), gram_matrix(style_feat.relu1_2))
                s2 = F.mse_loss(gram_matrix(stylized_feat.relu2_2), gram_matrix(style_feat.relu2_2))
                s3 = F.mse_loss(gram_matrix(stylized_feat.relu3_3), gram_matrix(style_feat.relu3_3))
                s4 = F.mse_loss(gram_matrix(stylized_feat.relu4_3), gram_matrix(style_feat.relu4_3))
                loss_style = (s1 + s2 + s3 + s4) * cfg["style_lambda"]
            else:
                loss_style = torch.zeros_like(loss_tv)

            # total loss
            total_loss = loss_tv + loss_l1 + loss_perceptual + loss_dist + loss_style

            optimizer_model.zero_grad()
            # optimizer_weight.zero_grad()
            total_loss.backward(torch.ones_like(total_loss))
            # optimizer_weight.step()
            optimizer_model.step()

            train_loss += total_loss.item()
            loss_tv_all += loss_tv.sum().item()
            loss_l1_all += loss_l1.sum().item()
            loss_percep_all += loss_perceptual.sum().item()
            loss_style_all += loss_style.sum().item()
            loss_dist_all += loss_dist.sum().item()
            imgs_all += face_imgs.size(0)

            if idx + 1 == epoch_length:
                train_losses_epoch.append(train_loss / imgs_all)

        vutils.save_image(style_attack_mask, saved_path)

        # save models
        # state = {
        #     'model': attack_model, 'weights': weights,
        #     'temperature': cfg["temperature"], 'epoch': epoch
        # }
        # prefix = f'{save_folder}/{cfg["backbone_name"]}_{cfg["head_name"]}_{cfg["target_identity"]}_data400_'
        # suffix = f'_dist{cfg["dist_lambda"]}_l1{cfg["l1_lambda"]}_t{cfg["temperature"]}_{cfg["device_name"]}'
        # saved_path = prefix + suffix + f'_epoch{epoch}.t7'
        # torch.save(state, saved_path)
        # test_str = test_stymask(cfg, saved_path)
        if early_stop(train_losses_epoch[-1]):
            print(test_stymask(cfg, saved_path))
            print('\n')
            break

        # scheduler_weight.step(train_losses_epoch[-1])
        scheduler_model.step(train_losses_epoch[-1])

        # epoch_time = time.time() - start_time
        # elapsed_time += epoch_time
        # ep_h, ep_m, ep_s = get_hms(epoch_time)
        # el_h, el_m, el_s = get_hms(elapsed_time)

        # print(
        #     f'Epoch {epoch}, lr_m: {lr}, train_loss: {train_loss / imgs_all:.4f}, tv_loss: {loss_tv_all / imgs_all:.4f},'
        #     f' l1: {loss_l1_all / imgs_all:.4f}, perc: {loss_percep_all / imgs_all:.4f}, style: {loss_style_all / imgs_all:.4f}'
        #     f' dist:{loss_dist_all / imgs_all:.4f},'
        #     f' Epoch time: {int(ep_h):02d}:{int(ep_m):02d}:{int(ep_s):02d},'
        #     f' Elapsed time: {int(el_h):02d}:{int(el_m):02d}:{int(el_s):02d}, {weights}'
        # )
        # print(test_str)


def test_stymask(cfg, saved_path):
    fix_randon_seed(cfg['seed'])

    # load face model
    model_path = f'{cfg["face_model_path"]}/{cfg["backbone_name"]}_{cfg["head_name"]}.pth'
    checkpoint = torch.load(model_path)
    face_model = checkpoint['backbone']
    face_model = face_model.eval().to(cfg["device"])

    # BN
    face_bn = Batch_Normalization(mean=cfg["img_mean"], std=cfg["img_std"], device=cfg["device"])

    # data
    batch_size = 1
    my_loader = get_MyData(batch_size=batch_size, mode=cfg["test_set"], pin_memory=False,
                           num_workers=4, target_identity=cfg["target_identity"], shuffle=False)

    # target face & target embedding
    target_transform = transforms.Compose([
        transforms.Resize(cfg["img_size"]),
        transforms.ToTensor(),
    ])
    target_path = f'{cfg["data_path"]}/{cfg["target_identity"]}/{cfg["target_img"]}'
    target_face_img = target_transform(Image.open(target_path)).unsqueeze(0).to(cfg["device"])

    face_landmark_detector = utils2.get_landmark_detector(
        landmark_detector_type='mobilefacenet', device=cfg["device"]
    )
    location_extractor = LandmarkExtractor(
        cfg["device"], face_landmark_detector, (112, 112)
    ).to(cfg["device"])
    fxz_projector = FaceXZooProjector(
        device=cfg["device"], img_size=(112, 112), patch_size=(112, 112)
    ).to(cfg["device"])

    # advmask
    uv_mask = transforms.ToTensor()(Image.open('./prnet/new_uv.png').convert('L'))

    orimask_path = f'./data/ori_styles/{cfg["style_selection"]}.jpg'
    orimask = transforms.ToTensor()(Image.open(orimask_path).convert('RGB'))
    orimask = orimask * uv_mask

    # advmask_path = f'./data/trained_mask/saved_stymask/' \
    #                f'{cfg["target_identity"]}_{cfg["backbone_name"]}_{cfg["head_name"]}.png'
    stymask = transforms.ToTensor()(Image.open(saved_path).convert('RGB'))
    total, attack_same_count, ori_same_count, total_ssim = 0, 0, 0, 0.
    target_embedding_ori = l2_norm(face_model(face_bn.norm(target_face_img)))
    for idx, [face_imgs, _, _] in enumerate(my_loader):
        face_imgs, stymask, orimask = face_imgs.to(cfg["device"]), stymask.to(cfg["device"]), orimask.to(cfg["device"])
        preds = location_extractor(face_imgs)
        ori_faces = fxz_projector(face_imgs, preds, orimask, do_aug=True)
        adv_faces = fxz_projector(face_imgs, preds, stymask, do_aug=True)


        target_embedding = target_embedding_ori.repeat(face_imgs.size(0), 1)

        ori_embedding = l2_norm(face_model(face_bn.norm(ori_faces)))
        diff = torch.subtract(ori_embedding, target_embedding)
        ori_dist = torch.sum(torch.square(diff), dim=1)
        ori_same_count += 1 if ori_dist < cfg["dist_threshold"] else 0

        attack_embedding = l2_norm(face_model(face_bn.norm(adv_faces)))
        diff = torch.subtract(attack_embedding, target_embedding)
        attack_dist = torch.sum(torch.square(diff), dim=1)
        attack_same_count += 1 if attack_dist < cfg["dist_threshold"] else 0

        total_ssim += compare_ssim(np.array(tensor2Image(adv_faces.squeeze(0))),
                                   np.array(tensor2Image(ori_faces.squeeze(0))),
                                   win_size=111, data_range=255, multichannel=True)
        total += face_imgs.size(0)

    adv_asr = attack_same_count / total
    ori_asr = ori_same_count / total
    print_str = f'{cfg["target_identity"]}\t{cfg["target_img"]}\t{ori_asr:.4f}\t{adv_asr:.4f}\t{total_ssim / total:.4f}'
    return print_str


def parseArgs():
    parser = argparse.ArgumentParser(description="Training Adv StyMask.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--device", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    if args.device == 0:
        # l_1
        l1_lambdas = [10, 100, 1000]
        for l1_lambda in l1_lambdas:
            cfg_l1_lambda = {'l1_lambda': l1_lambda}
            config = {
                **cfg_common, **cfg_targets['Vivica_Fox'],
                **cfg_devices['gpu0'], **cfg_l1_lambda,
                **cfg_sty_ab[1], **cfg_heads['ArcFace'],
            }
            train_stymask(config)

        # l_tv
        tv_lambdas = [1, 10, 100]
        for tv_lambda in tv_lambdas:
            cfg_tv_lambda = {'tv_lambda': tv_lambda}
            config = {
                **cfg_common, **cfg_targets['Vivica_Fox'],
                **cfg_devices['gpu0'], **cfg_tv_lambda,
                **cfg_sty_ab[1], **cfg_heads['ArcFace'],
            }
            train_stymask(config)
    else:
        # l_c
        percep_lambdas = [0.01, 0.001, 0.0001]
        for percep_lambda in percep_lambdas:
            cfg_percep_lambda = {'percep_lambda': percep_lambda}
            config = {
                **cfg_common, **cfg_targets['Vivica_Fox'],
                **cfg_devices['gpu1'], **cfg_percep_lambda,
                **cfg_sty_ab[1], **cfg_heads['ArcFace'],
            }
            train_stymask(config)

        # l_s
        style_lambdas = [100, 1000, 10000]
        for style_lambda in style_lambdas:
            cfg_style_lambda = {'style_lambda': style_lambda}
            config = {
                **cfg_common, **cfg_targets['Vivica_Fox'],
                **cfg_devices['gpu1'], **cfg_style_lambda,
                **cfg_sty_ab[1], **cfg_heads['ArcFace'],
            }
            train_stymask(config)
