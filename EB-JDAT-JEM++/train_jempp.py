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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch as t
import torch.nn as nn
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ExpUtils import *
from utils import eval_classification, Hamiltonian, checkpoint, get_data, set_bn_train, set_bn_eval, plot
from models.jem_models import get_model_and_buffer

t.set_num_threads(2)
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
inner_his = []
conditionals = []


def init_random(args, bs):
    global conditionals
    n_ch = 3
    size = [3, 32, 32]
    im_sz = 32
    new = t.zeros(bs, n_ch, im_sz, im_sz)
    for i in range(bs):
        index = np.random.randint(args.n_classes)
        dist = conditionals[index]
        new[i] = dist.sample().view(size)
    return t.clamp(new, -1, 1).cpu()


def sample_p_0(replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random(args, bs), []
    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
    inds = t.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(args, bs)
    choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(args.device), inds


def sample_q(f, replay_buffer, y=None, n_steps=10, in_steps=10, args=None, save=True):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    global inner_his
    inner_his = []
    # Batch norm uses train status
    # f.eval()
    # get batch size
    bs = args.batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    if in_steps > 0:
        Hamiltonian_func = Hamiltonian(f.f.layer_one)

    eps = args.eps
    if args.pyld_lr <= 0:
        in_steps = 0

    for it in range(n_steps):
        energies = f(x_k, y=y)
        e_x = energies.sum()
        # wgrad = f.f.conv1.weight.grad
        eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]
        # e_x.backward(retain_graph=True)
        # eta = x_k.grad.detach()
        # f.f.conv1.weight.grad = wgrad

        if in_steps > 0:
            p = 1.0 * f.f.layer_one_out.grad
            p = p.detach()

        tmp_inp = x_k.data
        tmp_inp.requires_grad_()
        if args.sgld_lr > 0:
            # if in_steps == 0: use SGLD other than PYLD
            # if in_steps != 0: combine outter and inner gradients
            # default 0
            tmp_inp = x_k + t.clamp(eta, -eps, eps) * args.sgld_lr
            tmp_inp = t.clamp(tmp_inp, -1, 1)

        for i in range(in_steps):

            H = Hamiltonian_func(tmp_inp, p)

            eta_grad = t.autograd.grad(H, [tmp_inp], only_inputs=True, retain_graph=True)[0]
            eta_step = t.clamp(eta_grad, -eps, eps) * args.pyld_lr

            tmp_inp.data = tmp_inp.data + eta_step
            tmp_inp = t.clamp(tmp_inp, -1, 1)

        x_k.data = tmp_inp.data

        if args.sgld_std > 0.0:
            x_k.data += args.sgld_std * t.randn_like(x_k)

    if in_steps > 0:
        loss = -1.0 * Hamiltonian_func(x_k.data, p)
        loss.backward()

    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0 and save:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples


def category_mean(dload_train, args):
    import time
    start = time.time()
    if args.dataset == 'svhn':
        size = [3, 32, 32]
    else:
        size = [3, 32, 32]
    centers = t.zeros([args.n_classes, int(np.prod(size))])
    covs = t.zeros([args.n_classes, int(np.prod(size)), int(np.prod(size))])

    im_test, targ_test = [], []
    for im, targ in dload_train:
        im_test.append(im)
        targ_test.append(targ)
    im_test, targ_test = t.cat(im_test), t.cat(targ_test)

    # conditionals = []
    for i in range(args.n_classes):
        imc = im_test[targ_test == i]
        imc = imc.view(len(imc), -1)
        mean = imc.mean(dim=0)
        sub = imc - mean.unsqueeze(dim=0)
        cov = sub.t() @ sub / len(imc)
        centers[i] = mean
        covs[i] = cov
        t.save(imc, f"data/img_class_{i}.pt")
    print(time.time() - start)
    
    t.save(centers, '%s_mean.pt' % args.dataset)
    t.save(covs, '%s_cov.pt' % args.dataset)


def init_from_centers(args):
    global conditionals
    from torch.distributions.multivariate_normal import MultivariateNormal
    bs = args.buffer_size
    if args.dataset == 'tinyimagenet':
        size = [3, 64, 64]
    elif args.dataset == 'svhn':
        size = [3, 32, 32]
    else:
        size = [3, 32, 32]
    centers = t.load('%s_mean.pt' % args.dataset)
    covs = t.load('%s_cov.pt' % args.dataset)

    buffer = []
    for i in range(args.n_classes):
        mean = centers[i].to(args.device)
        cov = covs[i].to(args.device)
        dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * t.eye(int(np.prod(size))).to(args.device))
        buffer.append(dist.sample((bs // args.n_classes, )).view([bs // args.n_classes] + size).cpu())
        conditionals.append(dist)
    return t.clamp(t.cat(buffer), -1, 1)



def normalize_tensor(in_feat, eps=1e-10):
    # print(in_feat.shape)
    # norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2))  # 直接计算L2范数
    return in_feat / (norm_factor + eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def feat_diff(outs0, outs1, retPerLayer=False):
    L = len(outs0)
    feats0, feats1, diffs = {}, {}, {}
    for kk in range(L):
        feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
        diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

    # Choose the device based on the first element of res or another tensor
    device = diffs[0].device  # Assuming diffs[0] is on the correct device
    
    res = []
    for kk in range(L):
        if diffs[kk].ndimension() == 1:
            res.append(diffs[kk].sum(dim=0, keepdim=True).to(device))  # Ensure it's on the correct device
        else:
            res.append(spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True).to(device))  # Ensure it's on the correct device

    val = t.zeros_like(res[0]).to(device)  # Ensure val is on the same device as res[0]
    for l in range(0, L):
        val += res[l]
        
    if retPerLayer:
        return val, res
    else:
        return val

def perceptionloss(f,x,x_adv):
    p_out_x = f.classify(x)
    p_out_xTilde = f.classify(x_adv)
    perception_distance = feat_diff(p_out_x,p_out_xTilde)
    return perception_distance.mean()

def xXTildeLoss(x, xTilde):
        #loss_drv = 0
        x_k = x.clone()
        x_tilde_k = xTilde.clone()

        loss_drv = t.nn.functional.mse_loss(x_k, x_tilde_k,reduction='mean')

        return loss_drv

def sampleXadv(f, x, y, device,in_steps=5, eps=8/255, alpha=2 / 255, args=None, save=True):
    """
    Parameters:
    - f: The model used for generating adversarial examples.
    - x: The input tensor (image) for which adversarial perturbations are to be generated.
    - y: The label tensor. Optional, used in classification tasks.
    - in_steps: The number of steps for the PGD method.
    - eps: The maximum perturbation size.
    - alpha: The step size for each perturbation step.
    - args: Additional arguments, such as learning rate, etc.
    - save: Boolean flag to save the generated adversarial example.
    """
    # Set model in evaluation mode
    f.eval()

    # Split input x into smaller batches
    batch_size = 2  # Assuming args.batch_size is defined elsewhere
    num_batches = (x.size(0) + batch_size - 1) // batch_size  # Calculate number of batches
    
    # Create a tensor to store the adversarial examples
    x_adv_all = []

    # Process each batch
    for batch_idx in range(num_batches):
        # Get the current batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, x.size(0))
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        # Make sure the input tensor requires gradient
        x_adv = x_batch.clone().detach()

        # Apply small random perturbation
        with t.no_grad():
            x_adv = x_adv + t.empty_like(x_adv).uniform_(-eps, eps)
            x_adv = t.clamp(x_adv, -1.0, 1.0)
        
        x_adv_k = t.autograd.Variable(x_adv, requires_grad=True).to(device)
        
        # Perform the attack for the current batch
        for k in range(in_steps):
            energies = f(x_adv_k,y_batch)
            e_x = energies.sum()
            p_loss = perceptionloss(f, x_batch, x_adv_k)  # Perception loss
            e_x = e_x + p_loss
            f_prime = t.autograd.grad(e_x, [x_adv_k],retain_graph=True)[0]
            # we also provide the version of using CrossEntropyLoss, which is more stable
            # loss = t.nn.CrossEntropyLoss()(f.classify(x_adv_k), y_batch)  # Cross-entropy loss
            # loss = loss +  p_loss  # Total loss
            # f_prime = t.autograd.grad(loss, [x_adv_k],retain_graph=True)[0]

            
            with t.no_grad():
                # Update the adversarial example
                x_adv_k.data += alpha * t.sign(f_prime.detach())
                x_adv_k.data = t.clamp(x_adv_k.data, -1.0, 1.0)

        # Collect the adversarial examples for this batch
        x_adv_all.append(x_adv_k.detach())

    # Combine all batches' adversarial examples
    x_adv_all = t.cat(x_adv_all, dim=0)

    f.train()  # Set model back to training mode

    # Return the adversarial example
    return x_adv_all

def main(args):

    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.seed)

    device = t.device(f'cuda:{args.gpu_id}' if t.cuda.is_available() else 'cpu')
    args.device = device


    f, replay_buffer = get_model_and_buffer(args, device)
 
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    # for dataset centers
    if not os.path.isfile('%s_cov.pt' % args.dataset):
        category_mean(dload_train, args)

    f, replay_buffer = get_model_and_buffer(args, device)
    if args.p_x_weight > 0:
        replay_buffer = init_from_centers(args)

    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0
    # trace learning rate
    new_lr = args.lr
    n_steps = args.n_steps
    in_steps = args.in_steps
    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))

        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)
            

            L = 0.
            if args.p_x_weight > 0:  # maximize log p(x)
                if args.plc == 'alltrain1':
                    fp_all = f(x_p_d)
                    fp = fp_all.mean()
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q, n_steps=n_steps, in_steps=in_steps, args=args)
                else:
                    x_q = sample_q(f, replay_buffer, n_steps=n_steps, in_steps=in_steps, args=args)  # sample from log-sumexp

                if args.plc == 'eval':
                    f.apply(set_bn_eval)
                    fp_all = f(x_p_d)
                    fp = fp_all.mean()
                if args.plc == 'alltrain2':
                    fp_all = f(x_p_d)
                    fp = fp_all.mean()
                fq_all = f(x_q)
                fq = fq_all.mean()

                l_p_x = -(fp - fq)
                if args.plc == 'eval':
                    f.apply(set_bn_train)

                if cur_iter % args.print_every == 0:
                    print('{} P(x) | {}:{:>d} f(x_p_d)={:>9.4f} f(x_q)={:>9.4f} d={:>9.4f}'.format(args.pid, epoch, i, fp, fq, fp - fq))

                L += args.p_x_weight * l_p_x

            if args.p_y_given_x_weight > 0:  # maximize log p(y | x)
                logits = f.classify(x_lab)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('{} P(y|x) {}:{:>d} loss={:>9.4f}, acc={:>9.4f}'.format(args.pid, epoch, cur_iter, l_p_y_given_x.item(), acc.item()))
                L += args.p_y_given_x_weight * l_p_y_given_x

            # if args.p_x_y_weight > 0:  # maximize log p(x, y)
            #     assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
            #     x_q_lab = sample_q(f, replay_buffer, y=y_lab, n_steps=n_steps, in_steps=in_steps, args=args)
            #     fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
            #     l_p_x_y = -(fp - fq)
            #     if cur_iter % args.print_every == 0:
            #         print('P(x, y) | {}:{:>d} f(x_p_d)={:>9.4f} f(x_q)={:>9.4f} d={:>9.4f}'.format(epoch, i, fp, fq, fp - fq))
            #
            #     L += args.p_x_y_weight * l_p_x_y

            if (args.p_x_xadv_weight > 0) or (args.p_y_given_xadv_weight > 0):  # maximize log p(x, xhead)
                
                assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                # x_q_lab = sample_q(f, replay_buffer, y=y_lab, n_steps=n_steps, in_steps=in_steps, args=args)
                # x_q = sample_q(f, replay_buffer, n_steps=n_steps, args=args)
                # logits = f.classify(x_lab)
                # ce_loss = nn.CrossEntropyLoss()(logits, y_lab)
                # if cur_iter % args.print_every == 0:
                #     acc = (logits.max(1)[1] == y_lab).float().mean()
                #     print('{} P(y|x) {}:{:>d} loss CE:{:>5.4f}, acc={:>5.4f}'.format(args.pid, epoch, cur_iter, ce_loss.item(), acc.item()))
                xadv_lab = sampleXadv(f, x=x_lab, y=y_lab, in_steps=in_steps, args=args)
                fp, fq = f(x_p_d).mean(), f(xadv_lab).mean()
                l_p_x_xadv = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print('P(xadv|x) | {}:{:>d} f(x_p)={:>9.4f} f(x_adv)={:>9.4f} d={:>9.4f}'.format(epoch, i, fp, fq, fp - fq))
                if args.p_x_xadv_weight > 0:
                    L += args.p_x_xadv_weight * l_p_x_xadv
                if args.p_y_given_xadv_weight > -1:  # maximize log p(y | xadv)
                    logits = f.classify(xadv_lab)
                    l_p_y_given_xadv = nn.CrossEntropyLoss()(logits, y_lab)
                    if cur_iter % args.print_every == 0:
                        robust = (logits.max(1)[1] == y_lab).float().mean()
                        print('{} P(y|xadv) {}:{:>d} loss={:>9.4f}, robust={:>9.4f}'.format(args.pid, epoch, cur_iter,
                                                                                            l_p_y_given_xadv.item(),
                                                                                            robust.item()))
                    L += args.p_y_given_xadv_weight * l_p_y_given_xadv
                    # print('loss={:>9.4f}'.format(L))



            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e8:
                print("BAD BOIIIIIIIIII")
                print("min {:>4.3f} max {:>5.3f}".format(x_q.min().item(), x_q.max().item()))
                plot('{}/diverge_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                return

            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1

            if cur_iter % args.print_every == 0 and args.p_x_weight > -1:
                if not args.plot_cond:
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, y=y_q, n_steps=n_steps, in_steps=in_steps, args=args)
                    else:
                        x_q = sample_q(f, replay_buffer, n_steps=n_steps, in_steps=in_steps, args=args)
                    plot('{}/samples/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, y=y, n_steps=n_steps, in_steps=in_steps, args=args)
                    plot('{}/samples/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)

        if epoch % args.ckpt_every == 0 and args.p_x_weight > 0:
            checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', args, device)

        if epoch % args.eval_every == 0 or (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
            f.eval()
            with t.no_grad():
                correct, loss = eval_classification(f, dload_valid, 'Valid', epoch, args, wlog)
                if args.dataset != 'tinyimagenet':
                    t_c, _ = eval_classification(f, dload_test, 'Test', epoch, args, wlog)
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Epoch {} Best Valid!: {}".format(epoch, correct))
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", args, device)
            f.train()
        checkpoint(f, replay_buffer, "last_ckpt.pt", args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LDA Energy Based Models")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100", 'tinyimagenet'])
    parser.add_argument("--data_root", type=str, default="./data/cifar10")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[60, 90, 120, 135], help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.2, help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=0.3)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_xadv_weight", type=float, default=1.)
    parser.add_argument("--p_x_xadv_weight", type=float, default=0.1)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "none", "batch", "instance", "layer", "act"], help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=10, help="number of steps of SGLD per iteration, 20 works for PCD")
    parser.add_argument("--in_steps", type=int, default=5, help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=0.05)

    # SGLD or PYLD
    parser.add_argument("--sgld_lr", type=float, default=0.0)
    parser.add_argument("--sgld_std", type=float, default=0)
    parser.add_argument("--pyld_lr", type=float, default=0.2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--dir_path", type=str, default='./experiment')
    parser.add_argument("--log_dir", type=str, default='./runs')
    parser.add_argument("--log_arg", type=str, default='JEMPP-n_steps-in_steps-pyld_lr-norm-plc')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000)

    parser.add_argument("--plc", type=str, default="alltrain1", help="alltrain1, alltrain2, eval")

    parser.add_argument("--eps", type=float, default=1, help="eps bound")
    parser.add_argument("--model", type=str, default='yopo')
    parser.add_argument("--novis", action="store_true", help="")
    parser.add_argument("--debug", action="store_true", help="")
    parser.add_argument("--exp_name", type=str, default="JEMPP", help="exp name, for description")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu-id", type=str, default="0")
    args = parser.parse_args()
    init_env(args, logger)
    args.save_dir = args.dir_path
    os.makedirs('{}/samples'.format(args.dir_path))
    print = wlog
    print(args.dir_path)
    main(args)
    print(args.dir_path)


