import os
import torch as t
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


def norm_ip(img, min, max):
    temp = t.clamp(img, min=min, max=max)
    temp = (temp + -min) / (max - min + 1e-5)
    return temp
from PIL import Image
import os
import numpy as np

def save_images_to_folder(feed_imgs, folder_path):
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 保存图像
    for i, img in enumerate(feed_imgs):
        # 将图像像素值范围从 [0, 255] 转换为 uint8 类型
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 转换为PIL图像对象
        pil_img = Image.fromarray(img)
        
        # 构造图像文件名
        image_filename = os.path.join(folder_path, f"image_{i+1}.png")
        
        # 保存图像
        pil_img.save(image_filename)
        print(f"Image {i+1} saved at {image_filename}")

# 示例：保存 feed_imgs 到指定文件夹



def eval_fid(f, device,replay_buffer, args):

    
    
    if isinstance(replay_buffer, list):
        images = replay_buffer[0]
    elif isinstance(replay_buffer, tuple):
        images = replay_buffer[0]
    else:
        images = replay_buffer

    feed_imgs = []
    for i, img in enumerate(images):
        n_img = norm_ip(img, -1, 1)
        new_img = n_img.cpu().numpy().transpose(1, 2, 0) * 255
        feed_imgs.append(new_img)

    feed_imgs = np.stack(feed_imgs)

    if 'cifar100' in args.dataset:
        from Task.data import Cifar100
        test_dataset = Cifar100(args, augment=False)
    elif 'cifar' in args.dataset:
        from Task.data import Cifar10
        test_dataset = Cifar10(args, full=True, noise=False)
    elif 'svhn' in args.dataset:
        from Task.data import Svhn
        test_dataset = Svhn(args, augment=False)
    else:
        assert False, 'dataset %s' % args.dataset
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=False)

    test_ims = []

    def rescale_im(im):
        return np.clip(im * 256, 0, 255).astype(np.uint8)
    
    for d in test_dataloader:

        if args.dataset == 'stl10':
            data = d[0].numpy().transpose(0, 2, 3, 1)
        else:
            data_corrupt, data, label_gt = d
            if args.dataset in ['celeba128', 'img32']:
                data = data_corrupt.numpy().transpose(0, 2, 3, 1)
            else:
                data = data.numpy()
        test_ims.extend(list(rescale_im(data)))
        if (args.dataset == "imagenet" or 'img' in args.dataset) and len(test_ims) > 60000:
            test_ims = test_ims[:60000]
            break

    fid = -1
    print(feed_imgs.shape, len(test_ims), test_ims[0].shape)
    test_imgs_np = np.stack(test_ims, axis=0)  # 得到形状 (60000, 32, 32, 3)
    
    test_imgs_tensor = t.from_numpy(test_imgs_np)
    feed_imgs_tensor = t.from_numpy(feed_imgs)
    test_imgs_tensor = test_imgs_tensor.permute(0, 3, 1, 2).to(t.uint8)  # 形状变为 (60000, 3, 32, 32)
    feed_imgs_tensor = feed_imgs_tensor.permute(0, 3, 1, 2).to(t.uint8) # 形状变为 (8995, 3, 32, 32)

    # for data_corrupt, data, label_gt in tqdm(test_dataloader):
    #     data = data.numpy()
    #     test_ims.extend(list(rescale_im(data)))

    # FID score
    # n = min(len(images), len(test_ims))
    # save_images_to_folder(test_ims, "./saved_images_cifar10")
    # from Task.fid import get_fid_score
    # fid = get_fid_score(feed_imgs,test_ims,device,dims=2048)
    # print("FID of score {}".format(fid))
    start = time.time()
            # fid = get_fid_score(feed_imgs, test_ims)
    from Task.fid import compute_fid
    fid= compute_fid(test_imgs_tensor,feed_imgs_tensor,device)
    from Task.fid import compute_is
    score, std = compute_is(feed_imgs_tensor, device)
            
    fid= compute_fid(test_imgs_tensor,feed_imgs_tensor,device)
    print("Inception score of {} with std of {} takes {}s".format(score, std, time.time() - start))
    print("FID of score {} takes {}s".format(fid, time.time() - start))
    return fid
