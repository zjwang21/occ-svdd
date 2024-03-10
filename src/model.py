import torch
import torch.nn as nn
import torch.nn.functional as F

class Occ_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rep_dim = 256
        self.fc1 = nn.Linear(4096, 2048, bias=False)
        self.ln1 = nn.LayerNorm(2048, eps=1e-12)
        self.fc2 = nn.Linear(2048, 768, bias=False)
        self.ln2 = nn.LayerNorm(768, eps=1e-12)
        self.fc3 = nn.Linear(768, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.ln1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.ln2(x))
        x = self.fc3(x)
        return x


class Occ_Net_Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.rep_dim = 256
        self.fc1 = nn.Linear(4096, 2048, bias=False)
        self.ln1 = nn.LayerNorm(2048, eps=1e-12)
        self.fc2 = nn.Linear(2048, 768, bias=False)
        self.ln2 = nn.LayerNorm(768, eps=1e-12)
        self.fc3 = nn.Linear(768, self.rep_dim, bias=False)

        # Decoder
        self.defc1 = nn.Linear(self.rep_dim, 768, bias=False,)
        self.ln3 = nn.LayerNorm(768, eps=1e-12)
        self.defc2 = nn.Linear(768, 2048, bias=False)
        self.ln4 = nn.LayerNorm(2048, eps=1e-12)
        self.defc3 = nn.Linear(2048, 4096, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.ln1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.ln2(x))
        x = self.fc3(x)

        x = self.defc1(x)
        x = F.leaky_relu(self.ln3(x))
        x = self.defc2(x)
        x = F.leaky_relu(self.ln4(x))
        x = self.defc3(x)
        x = torch.sigmoid(x)
        return x
