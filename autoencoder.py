import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Autoencoder(nn.Module):

    def __init__(self, btl_nck_dim=32, nc=3):
        super(Autoencoder, self).__init__()

        self.btl_nck_dim = btl_nck_dim
        self.nc = nc
        # Input :  # B,  3, 224, 224
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 112, 112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 56, 56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  28, 28
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(),
            View((-1, 512*28*28)),                                 # B, 512*25*25
        )

        self.fc = nn.Linear(512*28*28, btl_nck_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(btl_nck_dim, 512*28*28),                           # B, 512*14*14
            View((-1, 512, 28, 28)),                               # B, 512,  14,  14
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),   # B,  256, 28, 28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 56, 56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, nc, 4, 2, 1, bias=False),    # B,  nc, 112, 112
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self.encoder(x)
        latent_feat = self.fc(z)
        x_recon = self.decoder(latent_feat)
        return x_recon
