import os.path

from utils import fix_randon_seed
from mask_face.mask_functions import mask_face_img
import random
import cv2
from PIL import Image
import numpy as np


def choose_data(target_name):
    saved_folder = f'./data/target_attack_path/{target_name}'

    if os.path.exists(saved_folder):
        train_path =f'{saved_folder}/lfw_400_train_{target_name}.txt'
        test_path =f'{saved_folder}/lfw_1000_test_{target_name}.txt'
        if os.path.exists(train_path) and os.path.exists(test_path):
            f = open(train_path)
            train_data_num = len(f.readlines())  # 直接将文件中按行读到list里
            f.close()  # 关闭文件
            f = open(test_path)
            test_data_num = len(f.readlines())  # 直接将文件中按行读到list里
            f.close()  # 关闭文件
            if train_data_num == 400 and test_data_num == 1000:
                return
    else:
        os.mkdir(saved_folder)

    # read all the images
    # img_paths = []
    # for line in open('lfw_bcolz_data_paths.txt'):
    #     img_path = line.replace('\n', '')
    #     img_paths.append(img_path)
    img_paths = []
    for i, j, k in os.walk('./data/lfw_resume'):
        # print(i, k)
        for s in range(len(k)):
            img_paths.append(f'{i}/{k[s]}')

    # choose the training data
    i, train_num, train_paths = 0, 400, []
    f = open(f"{saved_folder}/lfw_{train_num}_train_{target_name}.txt", "w+")

    while i < train_num:
        rand_idx = random.randint(0, len(img_paths) - 1)
        try:
            img_path = img_paths[rand_idx]
        except IndexError:
            img_path = img_paths[rand_idx - 1]
            # img_path = img_paths[rand_idx]
        if target_name not in img_path:
            if img_paths[rand_idx] not in train_paths:
                img_Image = Image.open(img_paths[rand_idx]).convert('RGB')
                img_cv2 = np.array(img_Image)
                img_test = img_cv2[:, :, ::-1].copy()
                # img_test = cv2.imread(img_paths[rand_idx])
                # print(img_paths[rand_idx])
                _, _, _, face_num_test = mask_face_img(img_test)
                if face_num_test == 1:
                    if i == train_num - 1:
                        f.write(img_paths[rand_idx])
                    else:
                        f.write(img_paths[rand_idx] + "\n")
                    # print(img_paths[rand_idx])
                    train_paths.append(img_paths[rand_idx])
                    i += 1
    f.close()
    # choose the test data
    j, test_num, test_paths = 0, 1000, []
    f = open(f"{saved_folder}/lfw_{test_num}_test_{target_name}.txt", "w+")
    while j < test_num:
        rand_idx = random.randint(0, len(img_paths) - 1)
        if target_name not in img_paths[rand_idx]:
            if img_paths[rand_idx] not in train_paths:
                if img_paths[rand_idx] not in test_paths:
                    img_Image = Image.open(img_paths[rand_idx]).convert('RGB')
                    img_cv2 = np.array(img_Image)
                    img_test = img_cv2[:, :, ::-1].copy()
                    # img_test = cv2.imread(img_paths[rand_idx])
                    # print(img_paths[rand_idx])
                    _, _, _, face_num_test = mask_face_img(img_test)
                    if face_num_test == 1:
                        # print(img_paths[rand_idx])
                        if j == test_num - 1:
                            f.write(img_paths[rand_idx])
                        else:
                            f.write(img_paths[rand_idx] + "\n")
                        test_paths.append(img_paths[rand_idx])
                        j += 1
    f.close()


def get_identities():
    # mul heads
    f = open('50targets.txt')
    identities = f.readlines()  # 直接将文件中按行读到list里
    f.close()  # 关闭文件
    identities = [x.strip() for x in identities]  # 去除list中的'\n'
    return identities


def choose_identities(num_identity=5):
    identity_names = []
    for line in open('lfw_bcolz_data_paths.txt'):
        img_path = line.replace('\n', '')
        identity_name = img_path.split('/')[3]
        identity_names.append(identity_name)

    raw_identities = []
    duplicate_identities = []
    count = 0
    for identity_name in identity_names:
        # print(identity_name)
        if identity_name not in raw_identities:
            raw_identities.append(identity_name)
        else:
            if identity_name not in duplicate_identities:
                duplicate_identities.append(identity_name)
                count += 1

    chosed_identities = []
    i = 0
    while i < num_identity:
        rand_idx = random.randint(0, len(duplicate_identities))
        if duplicate_identities[rand_idx] not in chosed_identities:
            chosed_identities.append(duplicate_identities[rand_idx])
            i += 1
    return chosed_identities


if __name__ == '__main__':
    # choose_data('Michael_Chang')
    num_identities = 1
    chosed_identities = choose_identities(num_identities)
    print(f'Chosed {num_identities} identities:\n {chosed_identities}')
    for chosed_identity in chosed_identities:
        choose_data(chosed_identity)
