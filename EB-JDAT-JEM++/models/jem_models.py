import torch as t
import torch.nn as nn
from models import wideresnet
import models
from models import wideresnet_yopo
from scipy.special import softmax
import numpy as np
import utils
import math
im_sz = 32
n_ch = 3


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10, model='wrn', args=None):
        super(F, self).__init__()
        # default, wrn
        self.norm = norm
        if model == 'yopo':
            self.f = wideresnet_yopo.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        else:
            self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def feature(self, x):
        penult_z = self.f(x, feature=True)
        return penult_z

    def forward(self, x, y=None):
        penult_z = self.f(x, feature=True)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x, feature=True)
        output = self.class_output(penult_z).squeeze()
        return output
    # def predict(self, x):        
    #     # if self.normalize == True:
    #     #     x = (x - self.mean) / self.std
    #     x = x.astype(np.float32)  
    #     n_batches = math.ceil(x.shape[0] / self.batch_size)

    #     logits_list = []

    #     with t.no_grad(): 
    #         for i in range(n_batches):
    #             if self.vit_extractor is not None:
    #                 x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
    #                 x_batch_torch = t.as_tensor(x_batch, device=t.device('cuda'))
    #                 if self.model_type=="vit_aaa":
    #                     logits = self.model(x_batch_torch).cpu().numpy()
    #                 else:
    #                     logits = self.model(x_batch_torch).logits.cpu().numpy()
    #             else:
    #                 x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
    #                 x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
    #                 logits = self.model(x_batch_torch).cpu().numpy()
    #             logits_list.append(logits)
    #     logits = np.vstack(logits_list)
    #     return logits
    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = utils.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        elif loss_type == 'margin_loss_calib': # for underconfidence
            n_cls = len(y[0])
            predicted_class =  np.argmax(logits, axis=1)
            predicted_onehot = utils.one_hot_encode_v2(predicted_class, n_cls=n_cls)
            
            preds_predicted_class = (logits * predicted_onehot).sum(1, keepdims=True)
            diff = preds_predicted_class - logits  
            diff[predicted_onehot] = np.inf  
            margin = diff.min(1, keepdims=True)
            loss = margin
            
        elif loss_type == 'margin_loss_rand_underconf': 
            n_cls = len(y[0])
            
            softmaxes = softmax(logits, axis=1)
            
            predicted_class =  np.argmax(softmaxes, axis=1)
            predicted_onehot = utils.one_hot_encode_v2(predicted_class, n_cls=n_cls)
            
            preds_predicted_class = (softmaxes * predicted_onehot).sum(1, keepdims=True)
            diff = preds_predicted_class - softmaxes  
            diff[predicted_onehot] = np.inf  
            margin = diff.min(1, keepdims=True)
            loss = margin
            
        elif loss_type == 'margin_loss_overconf': # for overconfidence
            loss = np.max(softmax(logits, axis=1), axis=1, keepdims=True)           
        else:
            raise ValueError('Wrong loss.')

        # Ensure that loss is a numpy array and extract the value if necessary
        if isinstance(loss, np.ndarray):
            loss_value = loss  # If loss is a numpy array, just use it
        elif isinstance(loss, torch.Tensor):
            loss_value = loss.item()  # Extract the value if it's a tensor
        else:
            raise ValueError("Unexpected loss type.")

        return loss_value.flatten()


 



class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10, model='wrn', args=None):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes, model=model, args=args)

    def forward(self, x, y=None):
        logits = self.classify(x)

        if y is None:
            v = logits.logsumexp(1)
            # print("log sum exp", v)
            return v
        else:
            return t.gather(logits, 1, y[:, None])


def init_random(args, bs):
    im_sz = 32
    if args.dataset == 'tinyimagenet':
        im_sz = 64
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def get_model_and_buffer(args, device):
    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes, model=args.model)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        # f = t.nn.DataParallel(f)
        replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)
    return f, replay_buffer
