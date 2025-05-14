# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import utils
import torch as t, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
import numpy as np
import wideresnet
import pdb
import torch
import matplotlib.pyplot as plt
import torchattacks

from tqdm import tqdm
# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3
n_classes = 10


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, 10)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z)


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None):
        super(CCF, self).__init__(depth, width, norm=norm)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def cycle(loader):
    while True:
        for data in loader:
            yield data


def init_random(bs):
    return t.FloatTensor(bs, 3, 32, 32).uniform_(-1, 1)


def sample_p_0(device, replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random(bs), []
    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // n_classes
    inds = t.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
        assert not args.uncond, "Can't drawn conditional samples without giving me y"
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs)
    choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(args, device, f, replay_buffer, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    bs = args.batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(device, replay_buffer, bs=bs, y=y)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)
    x_k = x_k.to(device) 
    # sgld
    for k in range(args.n_steps):
        f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
        x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)
        
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples


def uncond_samples(f, args, device, save=True):
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    replay_buffer = t.FloatTensor(args.buffer_size, 3, 32, 32).uniform_(-1, 1)
    for i in range(args.n_sample_steps):
        samples = sample_q(args, device, f, replay_buffer)
        if i % args.print_every == 0 and save:
            plot('{}/samples_{}.png'.format(args.save_dir, i), samples)
        print(i)
    return replay_buffer


def cond_samples(f, replay_buffer, args, device, fresh=False):
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    if fresh:
        replay_buffer = uncond_samples(f, args, device, save=False)
    n_it = replay_buffer.size(0) // 100
    all_y = []
    for i in range(n_it):
        x = replay_buffer[i * 100: (i + 1) * 100].to(device)
        y = f.classify(x).max(1)[1]
        all_y.append(y)

    all_y = t.cat(all_y, 0)
    each_class = [replay_buffer[all_y == l] for l in range(10)]
    print([len(c) for c in each_class])
    for i in range(100):
        this_im = []
        for l in range(10):
            this_l = each_class[l][i * 10: (i + 1) * 10]
            this_im.append(this_l)
        this_im = t.cat(this_im, 0)
        if this_im.size(0) > 0:
            plot('{}/samples_{}.png'.format(args.save_dir, i), this_im)
        print(i)


def logp_hist(f, args, device):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    plt.switch_backend('agg')
    def sample(x, n_steps=args.n_steps):
        x_k = t.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples
    def grad_norm(x):
        x_k = t.autograd.Variable(x, requires_grad=True)
        f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)
    def score_fn(x):
        if args.score_fn == "px":
            return f(x).detach().cpu()
        elif args.score_fn == "py":
            return nn.Softmax()(f.classify(x)).max(1)[0].detach().cpu()
        elif args.score_fn == "pxgrad":
            return -t.log(grad_norm(x).detach().cpu())
        elif args.score_fn == "refine":
            init_score = f(x)
            x_r = sample(x)
            final_score = f(x_r)
            delta = init_score - final_score
            return delta.detach().cpu()
        elif args.score_fn == "refinegrad":
            init_score = -grad_norm(x).detach()
            x_r = sample(x)
            final_score = -grad_norm(x_r).detach()
            delta = init_score - final_score
            return delta.detach().cpu()
        elif args.score_fn == "refinel2":
            x_r = sample(x)
            norm = (x - x_r).view(x.size(0), -1).norm(p=2, dim=1)
            return -norm.detach().cpu()
        else:
            return f.classify(x).max(1)[0].detach().cpu()
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * t.randn_like(x)]
    )
    datasets = {
        "cifar10": tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False),
        "svhn": tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test"),
        "cifar100":tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=False),
        "celeba": tv.datasets.ImageFolder(root="/scratch/gobi1/gwohl/CelebA/splits",
                                          transform=tr.Compose([tr.Resize(32),
                                                                tr.ToTensor(),
                                                                tr.Normalize((.5, .5, .5), (.5, .5, .5)),
                                                                lambda x: x + args.sigma * t.randn_like(x)]))
    }

    score_dict = {}
    for dataset_name in args.datasets:
        print(dataset_name)
        dataset = datasets[dataset_name]
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4, drop_last=False)
        this_scores = []
        for x, _ in dataloader:
            x = x.to(device)
            scores = score_fn(x)
            print(scores.mean())
            this_scores.extend(scores.numpy())
        score_dict[dataset_name] = this_scores

    for name, scores in score_dict.items():
        plt.hist(scores, label=name, bins=100, normed=True, alpha=.5)
    plt.legend()
    plt.savefig(args.save_dir + "/fig.pdf")


def best_samples(f, replay_buffer, arg, device, fresh=False):
    energy_vector = []
    ratio = 0.9
    energy_vector_xy = []
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    replay_buffer = uncond_samples(f, arg, device, save=True)
    n_it = replay_buffer.size(0) // 100
    all_y = []
    all_px = []
    probs = []
    with t.no_grad():
        for i in tqdm(range(n_it)):
            x = replay_buffer[i * 100: (i + 1) * 100].to(device)
            logits = f.classify(x)
            y = logits.max(1)[1]
            px = logits.logsumexp(1)
            prob = nn.Softmax(dim=1)(logits).max(1)[0]
            all_y.append(y)
            probs.append(prob)
            all_px.append(px)

    
    all_y = t.cat(all_y, 0).to(replay_buffer.device)
    # all_y = t.cat(all_y, 0)
    probs = t.cat(probs, 0)
    print(probs.min().item())
    print((probs < 0).sum().item())
    all_px = t.cat(all_px, 0)
    print("%f %f %f" % (probs.mean().item(), probs.max().item(), probs.min().item()))
    each_class = [replay_buffer[all_y == l] for l in range(10)]
    each_class_probs = [probs[all_y == l] for l in range(10)]
    each_class_px = [all_px[all_y == l] for l in range(10)]
    print([len(c) for c in each_class])

    new_buffer = []
    ratio = abs(ratio)
    for c in range(10):
        each_probs = each_class_probs[c]
        # select
        each_metric = each_class_px[c]
        # each_metric = each_class_probs[c]

        if ratio < 1:
            topk = int(len(each_probs) * ratio)
        else:
            topk = int(ratio)
        topk = min(topk, len(each_probs))
        if ratio > 0:
            topks = t.topk(each_metric, topk, largest=ratio > 0)
            index_list = topks[1]
        else:
            topks = t.topk(each_metric, topk, largest=ratio > 0)
            index_list = topks[1]

        print('P(x) min %.3f max %.3f' % (-each_metric[index_list].max().item(), -each_metric[index_list].min().item()))
        print('Prob(y|x) max %.3f min %.3f' % (each_probs[index_list].max().item(), each_probs[index_list].min().item()))

        images = each_class[c].to(device)[index_list]
        new_buffer.append(images)
        plot('{}/topk_{}.png'.format(arg.save_dir, c), images)

    replay_buffer = t.cat(new_buffer, 0)
    print(replay_buffer.shape)


def test_clf(f, args, device):
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + t.randn_like(x) * args.sigma]
    )

    def sample(x, n_steps=args.n_steps):
        x_k = t.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    if args.dataset == "cifar_train":
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=True)
    elif args.dataset == "cifar_test":
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)
    elif args.dataset == "svhn_train":
        dset = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="train")
    else:  # args.dataset == "svhn_test":
        dset = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test")

    dload = DataLoader(dset, batch_size=100, shuffle=False, num_workers=4, drop_last=False)

    corrects, losses, pys, preds = [], [], [], []
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        if args.n_steps > 0:
            x_p_d = sample(x_p_d)
        logits = f.classify(x_p_d)
        py = nn.Softmax()(f.classify(x_p_d)).max(1)[0].detach().cpu().numpy()
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().detach().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        pys.extend(py)
        preds.extend(logits.max(1)[1].cpu().numpy())

    loss = np.mean(losses)
    correct = np.mean(corrects)
    t.save({"losses": losses, "corrects": corrects, "pys": pys}, os.path.join(args.save_dir, "vals.pt"))
    print(loss, correct)

def cond_is_fid(f, new_buffer, args, device, ratio=0.1, eval='all'):
    if isinstance(new_buffer, list):
        new_buffer = t.stack(new_buffer)  # Convert list of tensors to a single tensor
    elif not isinstance(new_buffer, t.Tensor):
        raise TypeError("new_buffer must be a tensor or list of tensors.")
    n_it = new_buffer.size(0) // 100
    all_y = []
    probs = []
    with t.no_grad():
        for i in range(n_it):
            x = new_buffer[i * 100: (i + 1) * 100].to(device)
            logits = f.classify(x)
            y = logits.max(1)[1]
            prob = nn.Softmax(dim=1)(logits).max(1)[0]
            all_y.append(y)
            probs.append(prob)

    all_y = t.cat(all_y, 0)
    probs = t.cat(probs, 0)
    new_buffer = new_buffer.to(device)

    each_class = [new_buffer[all_y == l] for l in range(args.n_classes)]
    each_class_probs = [probs[all_y == l] for l in range(args.n_classes)]
    print([len(c) for c in each_class])

    new_buffer = []
    for c in range(args.n_classes):
        each_probs = each_class_probs[c]
        # print("%d" % len(each_probs))
        if ratio < 1:
            topk = int(len(each_probs) * ratio)
        else:
            topk = int(ratio)
        topk = min(topk, len(each_probs))
        topks = t.topk(each_probs, topk)
        index_list = topks[1]
        images = each_class[c][index_list]
        new_buffer.append(images)

    new_buffer = t.cat(new_buffer, 0)
    print(new_buffer.shape)
    from Task.eval_buffer import eval_is_fid
    std, fid = eval_is_fid(new_buffer, device,args, eval=eval)
    if eval in ['fid', 'all',"gen"]:
        print("FID of score {}".format(fid))
    return std, fid

def PGD_attack(f, args, device):
    """
    Performs a PGD attack on the given model and saves the generated adversarial images.
    """
    # 初始化攻击
    f = f.to(device).eval()
    attacker = torchattacks.PGD(f, eps=8/255, alpha=1/255, steps=20, random_start=True)

    # 数据加载
    transform_test = tr.Compose([
        tr.ToTensor(),
        tr.Normalize((.5, .5, .5), (.5, .5, .5)),
        lambda x: x + torch.randn_like(x) * args.sigma
    ])
    
    # 选择数据集
    if args.dataset == "cifar_test":
        dset = tv.datasets.CIFAR10(root="./data/cifar10", transform=transform_test, download=True, train=False)
    elif args.dataset == "cifar100_train":
        dset = tv.datasets.CIFAR100(root="./data", transform=transform_test, download=True, train=True)
    elif args.dataset == "cifar100_test":
        dset = tv.datasets.CIFAR100(root="./data/cifar100", transform=transform_test, download=True, train=False)
    elif args.dataset == "svhn_train":
        dset = tv.datasets.SVHN(root="./data", transform=transform_test, download=True, split="train")
    elif args.dataset == "svhn_test":
        dset = tv.datasets.SVHN(root="./data", transform=transform_test, download=True, split="test")
    elif args.dataset == 'stl10':
        dset = tv.datasets.STL10(root="./data", transform=transform_test, download=True, split="test")
    else:
        dset = tv.datasets.CIFAR10(root="./data", transform=transform_test, download=True, train=False)

    dload = DataLoader(dset, batch_size=100, shuffle=False, num_workers=4 if not args.debug else 0)

    # 如果保存文件夹不存在，则创建它
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    corrects = []
    for idx, (x, y) in enumerate(tqdm(dload)):
        x, y = x.to(device), y.to(device)
        
        # 生成对抗样本
        adv_x = attacker(x, y)
        
        # 评估模型
        with torch.no_grad():
            logits = f.classify(adv_x)
            pred = logits.argmax(dim=1)
            correct = (pred == y).float().cpu().numpy()
            corrects.extend(correct)
            acc = np.mean(corrects) * 100
            print(f"Accuracy: {acc:.2f}%")

    accuracy = np.mean(corrects) * 100
    print(f"After PGD_Linf Attack Accuracy: {accuracy:.2f}%")
    return accuracy

def AA_attack(f, args, device):
    """
    Performs a AA attack on the given model.
    """
   
    atk = torchattacks.AutoAttack(f, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
    transform_test = tr.Compose([
        tr.ToTensor(),
        tr.Normalize((.5, .5, .5), (.5, .5, .5)),
        lambda x: x + t.randn_like(x) * args.sigma
    ])
    if args.dataset == "cifar_test":
        dset = tv.datasets.CIFAR10(root="./data/cifar10", transform=transform_test, download=True, train=False)
    elif args.dataset == "cifar100_train":
        dset = tv.datasets.CIFAR100(root="./data/cifar100", transform=transform_test, download=True, train=True)
    elif args.dataset == "cifar100_test":
        dset = tv.datasets.CIFAR100(root="./data/cifar100", transform=transform_test, download=True, train=False)
    elif args.dataset == "svhn_train":
        dset = tv.datasets.SVHN(root="./data", transform=transform_test, download=True, split="train")
    elif args.dataset == "svhn_test":
        dset = tv.datasets.SVHN(root="./data", transform=transform_test, download=True, split="test")
    elif args.dataset == 'stl10':
        dset = tv.datasets.STL10(root="./data", transform=transform_test, download=True, split="test")
    else:
        dset = tv.datasets.CIFAR10(root="./data", transform=transform_test, download=True, train=False)

    dload = DataLoader(dset, batch_size=192, shuffle=False, num_workers=4 if not args.debug else 0)

    corrects = []
    for x, y in tqdm(dload):
        x, y = x.to(device), y.to(device)
        # 生成对抗样本
        adv_x = atk(x, y)
        # 评估模型
        with torch.no_grad():
            logits = f.classify(adv_x)
            pred = logits.argmax(dim=1)
            correct = (pred == y).float().cpu().numpy()
            corrects.extend(correct)
            acc = np.mean(corrects) * 100
            print(f"Accuracy: {acc:.2f}%")
    
    accuracy = np.mean(corrects) * 100
    print(f"After AA Attack Accuracy: {accuracy:.2f}%")
    return accuracy

def main(args):
    utils.makedirs(args.save_dir)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm)
    print(f"loading model from {args.load_path}")

    # load em up
    ckpt_dict = t.load(args.load_path)
    f.load_state_dict(ckpt_dict["model_state_dict"])
    replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)


    if args.eval == "test_clf":
        test_clf(f, args, device)

    if args.eval == "cond_samples":
        cond_samples(f, replay_buffer, args, device, args.fresh_samples)

    if args.eval == "uncond_samples":
        uncond_samples(f, args, device)

    if args.eval == "gen":
       best_samples(f, replay_buffer, args, device, args.fresh_samples)

    if args.eval == "fid":
        _, fid = cond_is_fid(f, replay_buffer, args, device, ratio=0.9, eval='fid')

    if args.eval == "PGD":
        PGD_attack(f, args, device)

    if args.eval == "AA":
        AA_attack(f, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--eval", default="test_clf", type=str,
                        choices=["test_clf", "gen","fid","PGD","AA"])
    parser.add_argument("--score_fn", default="px", type=str,
                        choices=["px", "py", "pxgrad"], help="For OODAUC, chooses what score function we use.")
    parser.add_argument("--ood_dataset", default="svhn", type=str,
                        choices=["svhn", "cifar_interp", "cifar_100", "celeba"],
                        help="Chooses which dataset to compare against for OOD")
    parser.add_argument("--dataset", default="cifar_test", type=str,
                        choices=["cifar_train", "cifar_test", "svhn_test", "svhn_train"],
                        help="Dataset to use when running test_clf for classification accuracy")
    parser.add_argument("--datasets", nargs="+", type=str, default=[],
                        help="The datasets you wanna use to generate a log p(x) histogram")
    # optimization
    parser.add_argument("--batch_size", type=int, default=64)
    # regularization
    parser.add_argument("--sigma", type=float, default=3e-2)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"])
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=0)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='gen1')
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_sample_steps", type=int, default=100)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true")
    parser.add_argument("--fresh_samples", action="store_true",
                        help="If set, then we generate a new replay buffer from scratch for conditional sampling,"
                             "Will be much slower.")


    args = parser.parse_args()
    main(args)
