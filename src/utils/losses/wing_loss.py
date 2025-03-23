import torch
import torch.nn as nn

class WingLoss(nn.Module):
    def __init__(self, omega=10.0, epsilon=2.0):
        """
        Wing Loss as described in the paper: https://arxiv.org/abs/1711.06753
        :param omega: Width of the non-linear part.
        :param epsilon: Slope of the linear part.
        """
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = torch.abs(diff)
        flag = abs_diff < self.omega

        loss = flag * self.omega * torch.log(1 + abs_diff / self.epsilon) + \
               (~flag) * (abs_diff - self.C())

        return loss.mean()

    def C(self):
        value = 1 + self.omega / self.epsilon
        value_tensor = torch.tensor(value, dtype=torch.float32)
        return self.omega - self.omega * torch.log(value_tensor)

