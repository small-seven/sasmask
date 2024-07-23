import torch
import cv2
import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from mask_face.mask_functions import mask_face_img, cv2Image


def get_MyData(batch_size=64, mode='4', pin_memory=True, num_workers=4,
               target_identity='Aaron_Eckhart', shuffle=False):
    dataset = MyDataset(mode=mode, target_identity=target_identity)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=pin_memory, num_workers=num_workers)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, mode='lfw_400_train', target_identity='Target'):

        if 'lfw' in mode:
            self.file_list = f'./data/target_attack_path/{target_identity}/{mode}_{target_identity}.txt'
        else:
            self.file_list = f'./data/{mode}.txt'

        self.transform = transforms.Compose([
            transforms.Resize([112, 112]),
            transforms.ToTensor(),
        ])
        self.test_paths = []


        with open(self.file_list) as f:
            pairs = f.read().splitlines()[:]

        for i in range(len(pairs)):
            self.test_paths.append(pairs[i])

    def __getitem__(self, index):
        img_Image = Image.open(self.test_paths[index]).convert('RGB')
        img_cv2 = np.array(img_Image)
        img_test = img_cv2[:, :, ::-1].copy()
        masked_img_test_Image, mask_bin_Image, mask_Image, face_num_test = mask_face_img(img_test)

        if face_num_test != 1:
            print(f'test image contains {face_num_test} faces!')
            print(self.test_paths[index])

        img_test_Image = cv2Image(img_test)
        imglist = [img_test_Image, mask_bin_Image, mask_Image]

        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])
            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.test_paths)
