import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_ch=16, use_dropout=False):
        super().__init__()
        self.use_dropout = use_dropout
        C = base_ch

        # Encoder
        self.encoder1 = self._conv_block(in_channels, C)             # H x W
        self.encoder2 = self._conv_block(C, C*2, pool=True)          # H/2 x W/2
        self.encoder3 = self._conv_block(C*2, C*4, pool=True)        # H/4 x W/4
        self.encoder4 = self._conv_block(C*4, C*8, pool=True)        # H/8 x W/8

        # Bottleneck
        self.bottleneck = self._conv_block(C*8, C*16, pool=True)     # H/16 x W/16

        # Decoder
        self.decoder4 = self._upconv_block(C*16, C*8)                # H/8 x W/8
        self.decoder3 = self._upconv_block(C*8, C*4)                 # H/4 x W/4
        self.decoder2 = self._upconv_block(C*4, C*2)                 # H/2 x W/2
        self.decoder1 = self._upconv_block(C*2, C)                   # H x W

        self.final_conv = nn.Conv2d(C, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels, pool=False):
        layers = []
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        if self.use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bottleneck(e4)

        # Skip connections
        d4 = self.decoder4(b) + e4
        d3 = self.decoder3(d4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2) + e1

        out = self.final_conv(d1)
        return out

    def extract_features(self, x):
        # Return bottleneck features
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        b = self.bottleneck(e4)
        return b


class UNetLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, in_channels=3, out_channels=3, base_ch=16, use_dropout=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_channels=in_channels, out_channels=out_channels, base_ch=base_ch, use_dropout=use_dropout)
        self.mse = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, _, _ = batch
        recon = self(imgs)
        loss = self.mse(recon, imgs)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _, _ = batch
        recon = self(imgs)
        loss = self.mse(recon, imgs)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}