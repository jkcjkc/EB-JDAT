
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pytorch_image_generation_metrics

from pytorch_image_generation_metrics import get_fid
import torch


# compute KID
def compute_kid(
    real_images,
    fake_images,
    max_subset_size=10,
    device="cuda",
    feature="logits_unbiased",
    normalize=True,
):
    batch_size = 500

    kid = KernelInceptionDistance(
        feature=feature, subset_size=max_subset_size, normalize=normalize
    ).to(device=device)

    for i in range(0, real_images.size(0), batch_size):
        batch = real_images[i : i + batch_size].to(device)
        kid.update(batch, real=True)

    for i in range(0, fake_images.size(0), batch_size):
        batch = fake_images[i : i + batch_size].to(device)
        kid.update(batch, real=False)

    KID = kid.compute()

    return KID[0], KID[1]


def compute_is(imgs, device="cuda"):
    """
    Compute the Inception Score
    Imgs is a tensor of shape (N, C, H, W) with values in [0, 1]
    """
    batch_size = 512
    inception = InceptionScore(feature="logits_unbiased", normalize=True).to(device)

    for i in range(0, imgs.size(0), batch_size):
        batch = imgs[i : i + batch_size].to(device)
        inception.update(batch)
    IS = inception.compute()
    return IS[0], IS[1]


def compute_fid(real_images, fake_images, device="cuda", feature=2048, normalize=True):
    """
    All images will be resized to 299 x 299 which is the size of the original training data.
    The boolian flag real determines if the images should update the statistics of the real
    distribution or the fake distribution.
    real_images: tensor of shape (N, C, H, W) with values in [0, 1], IT SHOULD BE CIFAR-10 TRAIN SET or TEST SET
    fake_images: tensor of shape (M, C, H, W) with values in [0, 1]
    """
    ### compute the FID score with torchmetrics

    batch_size = 512
    fid = FrechetInceptionDistance(feature=feature, normalize=normalize).to(
        device=device
    )

    for i in range(0, real_images.size(0), batch_size):
        batch = real_images[i : i + batch_size].to(device)
        fid.update(batch, real=True)

    for i in range(0, fake_images.size(0), batch_size):
        batch = fake_images[i : i + batch_size].to(device)
        fid.update(batch, real=False)

    ### compute the FID score with python-gan-metrics
    # FID = get_fid(
    #     fake_images,
    #     real_images,
    # )

    return fid.compute()

def compute_lpips(train_loader, fake_images, device="cuda", net="alex"):
    """
    Compute the LPIPS score
    fake_images is a tensor of shape (N, C, H, W) with values in [0, 1]
    train loader is the dataloader of the training set
    """

    imgs, labels = [], []
    for i, (img, label) in enumerate(train_loader):
        imgs.append(img)
        labels.append(label)

    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)

    # sort the images for class, to compute the LPIPS
    imgs = imgs[torch.argsort(labels)]

    assert imgs.size() == fake_images.size()

    batch_size = 1000
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True).to(
        device=device
    )

    for i in range(0, imgs.size(0), batch_size):
        batch_real = imgs[i : i + batch_size].to(device)
        batch_fake = fake_images[i : i + batch_size].to(device)
        lpips.update(batch_real, batch_fake)

    return lpips.compute()

