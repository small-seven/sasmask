import torch
from parameters_dist import thresholds
import os

embedders_dict = {
    'resnet18': {
        'layers': [2, 2, 2, 2],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet18.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet18.pth')
            }
        }
    },
    'resnet34': {
        'layers': [3, 4, 6, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet34.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet34.pth')
            }
        }
    },
    'resnet50': {
        'layers': [3, 4, 14, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet50.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet50.pth')
            }
        }
    },
    'resnet100': {
        'layers': [3, 13, 30, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet100.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet100.pth')
            },
            'magface': {
                'weights_path': os.path.join('..', 'face_recognition', 'magface_torch', 'weights',
                                             'magface_resnet100.pth')
            }
        }
    }
}

cfg_common = dict(
    # random seed
    seed=123,

    # checkpoint path
    style_model_path='./checkpoint/style_models/style_model_epoch_0.pth',
    style_attack_path='./checkpoint/attack_models',
    face_model_path='./checkpoint/face_models',

    # data setting
    data_path='./data/lfw_resume',
    train_set='lfw_400_train',
    test_set='lfw_1000_test',
    img_size=112,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    vgg_mean=(0.485, 0.456, 0.406),
    vgg_std=(0.229, 0.224, 0.225),
    target_identity='Steve_Park',
    target_img='Steve_Park_0001.jpg',
    pin_memory=False,
    num_workers=8,

    # parameters
    tv_lambda=10,  # 10e0=0.11709
    l1_lambda=100,  # 10e1=0.92
    percep_lambda=0.001,  # 10e-3=0.46624
    style_lambda=1000,  # 10e3=0.82
    dist_lambda=100,  # 1 = 1.82

    # face model setting
    backbone_name='ResNet_50',
    head_name='ArcFace',
    dist_threshold=thresholds['ResNet_50']['ArcFace'],

    # training parameters
    batch_size=1,
    num_epoch=2000,
    model_lr=0.01,
    weight_lr=0.01,
    lr_step=50,
    lr_gamma=0.1,
    temperature=0.1,
)

cfg_sty_ab = {
    0: dict(
        style_selection=0,
    ),
    1: dict(
        style_selection=1,
    ),
    2: dict(
        style_selection=2,
    ),
    3: dict(
        style_selection=3,
    ),
    4: dict(
        style_selection=4,
    ),
    5: dict(
        style_selection=5,
    ),
    6: dict(
        style_selection=6,
    ),
    7: dict(
        style_selection=7,
    ),
    8: dict(
        style_selection=8,
    ),
    9: dict(
        style_selection=9,
    ),
}

cfg_loss_ab = {
    'TV': dict(
        loss_ablation='TV',
        tv_lambda=0,
    ),

    'L1': dict(
        loss_ablation='L1',
        l1_lambda=0,
    ),
    'TV_L1': dict(
        loss_ablation='TV_L1',
        tv_lambda=0,
        l1_lambda=0,
    ),
    'Content': dict(
        loss_ablation='Content',
        percep_lambda=0,

    ),
    'Style': dict(
        loss_ablation='Style',
        style_lambda=0,
    ),
    'Content_Style': dict(
        loss_ablation='Content_Style',
        percep_lambda=0,
        style_lambda=0,
    ),
    'TV_L1_S_C': dict(
        loss_ablation='TV_L1_S_C',
        tv_lambda=0,
        l1_lambda=0,
        percep_lambda=0,
        style_lambda=0,
    ),
}

cfg_backbones = {
    'ResNet_50': dict(
        backbone_name='ResNet_50',
        head_name='ArcFace',
        dist_threshold=thresholds['ResNet_50']['ArcFace'],  # 0.08017
        # tv_lambda=10e1,  # 10e1=1.4
        # l1_lambda=10e2,  # 10e2=4.9
        # percep_lambda=10e-3,  # 10e-2=4.6
        # style_lambda=10e3,  # 10e3=0.8
        # dist_lambda=10e2,  # 10e0=0.1
    ),
    'ResNet_34': dict(
        backbone_name='ResNet_34',
        head_name='ArcFace',
        dist_threshold=thresholds['ResNet_34']['ArcFace'],
        # tv_lambda=10,  # 10e1=1.02
        # l1_lambda=100,  # 10e2=5.6
        # percep_lambda=10e-2,  # 10e-2=4.5
        # style_lambda=1000,  # 10e3=0.84
        # dist_lambda=100,  # 10e0=0.238
    ),
    'ResNet_101': dict(
        backbone_name='ResNet_101',
        head_name='ArcFace',
        dist_threshold=thresholds['ResNet_101']['ArcFace'],  # 0.07973
        # tv_lambda=10,  # 10e1=1.4
        # l1_lambda=100,  # 10e2=6.3
        # percep_lambda=10e-3,  # 10e-2=4.8
        # style_lambda=10e3,  # 10e3=0.9
        # dist_lambda=10e2,  # 10e0=0.069
    ),
    'MobileFaceNet': dict(
        backbone_name='MobileFaceNet',
        head_name='ArcFace',
        dist_threshold=thresholds['MobileFaceNet']['ArcFace'],
        # dist_lambda=100,
    ),
    'GhostNet': dict(
        backbone_name='GhostNet',
        head_name='ArcFace',
        dist_threshold=thresholds['GhostNet']['ArcFace'],
        # dist_lambda=800,
    ),
}

cfg_heads = {
    'ArcFace': dict(
        backbone_name='ResNet_50',
        head_name='ArcFace',
        dist_threshold=thresholds['ResNet_50']['ArcFace'],

    ),
    'CosFace': dict(
        backbone_name='ResNet_50',
        head_name='CosFace',
        dist_threshold=thresholds['ResNet_50']['CosFace'],

    ),
    'CircleLoss': dict(
        backbone_name='ResNet_50',
        head_name='CircleLoss',
        dist_threshold=thresholds['ResNet_50']['CircleLoss'],

    ),
    'CurricularFace': dict(
        backbone_name='ResNet_50',
        head_name='CurricularFace',
        dist_threshold=thresholds['ResNet_50']['CurricularFace'],

    ),
    'MagFace': dict(
        backbone_name='ResNet_50',
        head_name='MagFace',
        dist_threshold=thresholds['ResNet_50']['MagFace'],
    ),
}

cfg_targets = {
    'Steve_Park': dict(
        target_identity='Steve_Park',
        target_img='Steve_Park_0001.jpg',
        # tv_lambda=1e1,  # 10e1=1.3
        # l1_lambda=1e2,  # 10e2=7.8
        # percep_lambda=1e-3,  # 10e-2=4.4
        # style_lambda=1e3,  # 10e3=1.4
        # dist_lambda=1e2,  # 10e0=0.103
    ),
    'Patricia_Hearst': dict(
        target_identity='Patricia_Hearst',
        target_img='Patricia_Hearst_0001.jpg',
        # tv_lambda=1e1,  # 10e1=3.6
        # l1_lambda=1e2,  # 10e2=4.9
        # percep_lambda=1e-3,  # 10e-2=5
        # style_lambda=1e3,  # 10e3=0.6
        # dist_lambda=1e2,  # 10e0=0.11
    ),
    'Vivica_Fox': dict(
        target_identity='Vivica_Fox',
        target_img='Vivica_Fox_0001.jpg',
        # tv_lambda=1e1,  # 10e1=1.4
        # l1_lambda=1e2,  # 10e2=4.9
        # percep_lambda=1e-3,  # 10e-2=4.6
        # style_lambda=1e3,  # 10e3=0.8
        # dist_lambda=1e2,  # 10e0=0.1
    ),
    'Aaron_Eckhart': dict(
        target_identity='Aaron_Eckhart',
        target_img='Aaron_Eckhart_0001.jpg',
        # tv_lambda=1e1,  # 10e1=1.2
        # l1_lambda=1e2,  # 10e2=7.8
        # percep_lambda=1e-3,  # 10e-2=4.4
        # style_lambda=1e3,  # 10e3=0.8
        # dist_lambda=1e2,  # 10e0=0.11
    ),
}

cfg_datasets = {
    'VGGFace2_FP': dict(
        dataset_road='./data/VGGFace2_FP_resume',
        test_set='VGGFace2_FP_1000_test',
        backbone_name='ResNet_50',  # ResNet_50 ResNet_18 ResNet_34 InceptionResnetV1
        head_name='ArcFace',  # ['vggface2', 'ArcFace', 'CosFace', 'SphereFace']
        dist_threshold=0.10877,
    ),
    'AgeDB': dict(
        dataset_road='./data/agedb_resume',
        test_set='AgeDB_1000_test',
        backbone_name='ResNet_50',  # ResNet_50 ResNet_18 ResNet_34 InceptionResnetV1
        head_name='ArcFace',  # ['vggface2', 'ArcFace', 'CosFace', 'SphereFace']
        dist_threshold=0.11227,
    ),
    'CFP': dict(
        dataset_road='./data/cfp_ff_resume',
        test_set='CFP_1000_test',
        backbone_name='ResNet_50',  # ResNet_50 ResNet_18 ResNet_34 InceptionResnetV1
        head_name='ArcFace',  # ['vggface2', 'ArcFace', 'CosFace', 'SphereFace']
        dist_threshold=0.09338,
    ),
}

cfg_devices = {
    'gpu0': dict(
        device_name='gpu0',
        device='cuda:0' if torch.cuda.is_available else 'cpu',
        pin_memory=False,
    ),
    'gpu1': dict(
        device_name='gpu1',
        device='cuda:1' if torch.cuda.is_available else 'cpu',
        pin_memory=False,
    ),
    'gpu2': dict(
        device_name='gpu2',
        device='cuda:2' if torch.cuda.is_available else 'cpu',
        pin_memory=False,
    ),
    'gpu3': dict(
        device_name='gpu3',
        device='cuda:3' if torch.cuda.is_available else 'cpu',
        pin_memory=False,
    ),
    'gpu4': dict(
        device_name='gpu4',
        device='cuda:4' if torch.cuda.is_available else 'cpu',
        pin_memory=False,
    ),
    'gpu5': dict(
        device_name='gpu5',
        device='cuda:5' if torch.cuda.is_available else 'cpu',
        pin_memory=False,
    ),
    'gpu6': dict(
        device_name='gpu6',
        device='cuda:6' if torch.cuda.is_available else 'cpu',
        pin_memory=False,
    ),
    'gpu7': dict(
        device_name='gpu7',
        device='cuda:7' if torch.cuda.is_available else 'cpu',
        pin_memory=False,
    ),
}
