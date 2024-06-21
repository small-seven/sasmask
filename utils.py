import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import torch
import os.path
import random
from mask_face.mask_functions import mask_face_img
from PIL import Image
import numpy as np


def choose_data(target_name):
    # read all the images
    img_paths = []
    for line in open('lfw_bcolz_data_paths.txt'):
        img_path = line.replace('\n', '')
        img_paths.append(img_path)

    # make the target identity folder
    saved_folder = f'./data/target_attack_path/{target_name}'
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)

    # choose the training data
    i, train_num, train_paths = 0, 400, []
    f = open(f"{saved_folder}/lfw_{train_num}_train_{target_name}.txt", "w+")

    while i < train_num:
        rand_idx = random.randint(0, len(img_paths))
        if target_name not in img_paths[rand_idx]:
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


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def gram_matrix(inputs):
    '''
    from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    '''
    a, b, c, d = inputs.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = inputs.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def loss_calc(content, style, pastiche, lambda_c, lambda_s):
    c = F.mse_loss(pastiche.relu4_3, content.relu4_3)

    s1 = F.mse_loss(gram_matrix(pastiche.relu1_2), gram_matrix(style.relu1_2))
    s2 = F.mse_loss(gram_matrix(pastiche.relu2_2), gram_matrix(style.relu2_2))
    s3 = F.mse_loss(gram_matrix(pastiche.relu3_3), gram_matrix(style.relu3_3))
    s4 = F.mse_loss(gram_matrix(pastiche.relu4_3), gram_matrix(style.relu4_3))
    #    print("Content:",c.data.item())
    #    print("Style:",s1.data.item()+s2.data.item()+s3.data.item()+s4.data.item())

    loss = lambda_c * c + lambda_s * (s1 + s2 + s3 + s4)

    return loss


class Batch_Normalization(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), device='cuda:0'):
        self.mean = torch.Tensor(mean).unsqueeze(-1).unsqueeze(-1).to(device)
        self.std = torch.Tensor(std).unsqueeze(-1).unsqueeze(-1).to(device)
        # print(self.mean)
        # print(self.mean.size())
        # print(self.std)
        # print(self.std.size())

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def unnorm(self, tensor):
        return tensor * self.std + self.mean


def get_mask_location(img):
    # img: 4D tensor images [1,3,h,w]
    xs = []
    ys = []
    for i in range(img.size(2)):
        if img[:, :, i, :].detach().sum().item() > 0:
            xs.append(i)
    for j in range(img.size(3)):
        if img[:, :, :, j].detach().sum().item() > 0:
            ys.append(j)
    x_start, x_end = xs[0], xs[len(xs) - 1]
    y_start, y_end = ys[0], ys[len(ys) - 1]
    x_diff = x_end - x_start
    y_diff = y_end - y_start
    win = x_diff if x_diff < y_diff else y_diff
    return img[:, :, x_start:x_end + 1, y_start:y_end + 1], win


def gram_matrix(inputs):
    '''
    from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    '''
    a, b, c, d = inputs.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = inputs.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


# generate circle patch mask
def generate_circle_patch_mask(imgs_size=112, radius=10):
    x_center = 90
    y_center = 56
    patch_mask = torch.zeros(1, 3, imgs_size, imgs_size)
    for i in range(imgs_size):
        for j in range(imgs_size):
            distance = (i - x_center) ** 2 + (j - y_center) ** 2
            if distance < radius ** 2:
                patch_mask[:, :, i, j] = 1.
    # plt.imshow(tensor2Image(patch_mask.squeeze(0)))
    # plt.show()
    return patch_mask, radius


def fix_randon_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置python哈希种子，为了禁止hash随机化
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus
    cudnn.benchmark = False
    cudnn.deterministic = True


def compare_psnr(img1, img2, maxvalue=255):
    '''
    一般 opencv 读图是 uint8, 漏掉转数据格式会导致计算出错;
    有些代码会在 mse==0 的时候直接返回 100, 但很明显 psnr 并没有最大值为 100 的说法,
    通常也不会算两张相同图像的 psnr, 干脆和 skimage 一样不写 mse==0 的情况
    用法：
    img1, img2 = Image.open(img_name1), Image.open(img_name2)
    psnr = compare_psnr(np.array(img1), np.array(img2))
    '''
    img1, img2 = np.array(img1), np.array(img2)
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10((maxvalue ** 2) / mse)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d}'


if __name__ == '__main__':
    bn = Batch_Normalization()
