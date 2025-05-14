import torch
import torch.nn as nn

from ..attack import Attack
from torch.nn import functional as F


class Cali(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("Cali", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        ori_images = images.clone().detach().to(self.device)


        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        m = torch.nn.Softmax(dim=-1)

        with torch.no_grad():
            temp_output = self.get_logits(images )

        _, clean_predictions = torch.max(F.softmax(temp_output, dim=1), dim=1)



        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=-1, max=1).detach()

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.get_logits(images)
            self.model.zero_grad()
            cost = loss(outputs, labels).cuda()
            cost.backward()
            adv_images = images - self.alpha*images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            eta =  F.dropout(eta, p=0.1)
            images_temp = torch.clamp(ori_images + eta, min=-1, max=1).detach_()
            with torch.no_grad():
                outputs2 = self.get_logits(images_temp )

            softmaxes_ece = F.softmax(outputs2, dim=1)
            _, predictions_ece = torch.max(softmaxes_ece, dim=1)
            label_unchanged = torch.eq(predictions_ece , clean_predictions)
            images = images.detach_()
            images[label_unchanged] = images_temp[label_unchanged]




            # adv_images.requires_grad = True
            # outputs = self.get_logits(adv_images)

            # # Calculate loss
            # if self.targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            #     cost = loss(outputs, labels)

            # # Update adversarial images
            # grad = torch.autograd.grad(
            #     cost, adv_images, retain_graph=False, create_graph=False
            # )[0]

            # adv_images = adv_images.detach() + self.alpha * grad.sign()
            # delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        return images
