import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    Residual3DConvBlock,
    _sn_to_specnorm,
    _facify,
    GeneratorCombined1Block,
    GeneratorCombined2Block,
)

from util import compute_same_padding


class Encoder(nn.Module):
    def __init__(
        self,
        width,
        height,
        depth,
        in_channels,
        latent_dim,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(128, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = Residual3DConvBlock(
            in_channels=_facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=1,
            sn=sn,
            device=device,
        )
        self.step4 = Residual3DConvBlock(
            in_channels=_facify(128, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )
        self.step5 = Residual3DConvBlock(
            in_channels=_facify(256, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=1,
            sn=sn,
            device=device,
        )
        self.step6 = Residual3DConvBlock(
            in_channels=_facify(256, fac),
            n_filters=_facify(512, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )
        self.step7 = specnorm(
            nn.Conv3d(
                in_channels=_facify(512, fac),
                out_channels=_facify(512, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step8 = nn.GroupNorm(1, _facify(512, fac))
        self.step9 = nn.LeakyReLU()
        self.to_latent_mu = specnorm(
            nn.Linear(
                in_features=(width // 4)
                * (height // 4)
                * (depth // 4)
                * _facify(512, fac),
                out_features=latent_dim,
            )
        )
        self.to_latent_logvar = specnorm(
            nn.Linear(
                in_features=(width // 4)
                * (height // 4)
                * (depth // 4)
                * _facify(512, fac),
                out_features=latent_dim,
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)  # torch.Size([32, 1, 16, 16, 16])
        out = self.step2(out)  # torch.Size([32, 1, 16, 16, 16])
        out = self.step3(out)  # torch.Size([32, 1, 16, 16, 16])
        out = self.step4(out)  # torch.Size([32, 3, 8, 8, 8])
        out = self.step5(out)  # torch.Size([32, 3, 8, 8, 8])
        out = self.step6(out)  # torch.Size([32, 6, 4, 4, 4])
        out = self.step7(out)  # torch.Size([32, 6, 4, 4, 4])
        out = self.step8(out)  # torch.Size([32, 6, 4, 4, 4])
        out = self.step9(out)  # torch.Size([32, 6, 4, 4, 4])
        out = torch.flatten(out, start_dim=1, end_dim=-1)  # torch.Size([32, 384])
        mu = self.to_latent_mu(out)  # torch.Size([32, 1])
        logvar = self.to_latent_logvar(out)  # torch.Size([32, 1])
        return mu, logvar


class Encoder2(nn.Module):
    def __init__(
        self,
        width,
        height,
        depth,
        in_channels,
        latent_dim,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        encoder_modules = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(128, fac),
                    kernel_size=5,
                    stride=1,
                    padding=compute_same_padding(5, 1, 1),
                )
            ),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                in_channels=_facify(128, fac),
                n_filters=_facify(128, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(128, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(256, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            specnorm(
                nn.Conv3d(
                    in_channels=_facify(512, fac),
                    out_channels=_facify(512, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, _facify(512, fac)),
            nn.LeakyReLU(),
            nn.Flatten(),
        ]
        self.featurizer = nn.Sequential(*encoder_modules)
        num_features = (width // 4) * (height // 4) * (depth // 4) * _facify(512, fac)
        self.to_latent_mu = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )
        self.to_latent_logvar = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )

    def forward(self, inputs):
        features = self.featurizer(inputs)
        mu = self.to_latent_mu(features)  # torch.Size([32, 1])
        logvar = self.to_latent_logvar(features)  # torch.Size([32, 1])
        return mu, logvar


class Encoder3(nn.Module):
    def __init__(
        self,
        width,
        height,
        depth,
        in_channels,
        latent_dim,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)

        encoder_modules = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(512, fac),
                    kernel_size=5,
                    stride=1,
                    padding=compute_same_padding(5, 1, 1),
                )
            ),
            nn.GroupNorm(1, _facify(512, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            nn.Flatten(),
        ]
        self.featurizer = nn.Sequential(*encoder_modules)
        # num_features = width * height * depth
        num_features = (width // 4) * (height // 4) * (depth // 4) * _facify(512, fac)
        # num_features = (width // 8) * (height // 8) * (depth // 8) * _facify(1024, fac)
        self.to_latent_mu = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )
        self.to_latent_logvar = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )

    def forward(self, inputs):
        features = self.featurizer(inputs)
        mu = self.to_latent_mu(features)
        logvar = self.to_latent_logvar(features)
        return mu, logvar


class EmbedNoise(nn.Module):
    def __init__(self, z_dim, channels, dim=4, sn=0):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.pad = nn.Linear(z_dim, channels * dim * dim * dim)
        self.pad = specnorm(self.pad)
        # self.pad = nn.ConstantPad3d(padding=(3, 3, 3, 3, 3, 3), value=0.)  # -> (B, z_dim, 7, 7, 7)
        # self.conv = nn.Conv3d(z_dim, channels, kernel_size=4, stride=1, padding=0)  # -> (B, channels, 4, 4, 4)
        self.nonlin = nn.LeakyReLU()
        self.z_dim = z_dim
        self.channels = channels
        self.dim = dim

    def forward(self, z):
        # batch_size = z.shape[0]
        out = self.pad(z)
        # out = self.conv(out.view((-1, self.z_dim, 7, 7, 7)))
        out = self.nonlin(out)
        out = out.view((-1, self.channels, self.dim, self.dim, self.dim))
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac=1,
        testmode=False,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(128, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(128, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond1 = Residual3DConvBlock(
            _facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )
        self.downsample_cond2 = Residual3DConvBlock(
            _facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise = EmbedNoise(z_dim, _facify(128, fac), sn=sn)

        self.embed_noise_in = EmbedNoise(z_dim, 1, dim=16, sn=sn)
        self.combine_in_noise = GeneratorCombined1Block(
            _facify(128, fac) + 1, _facify(128, fac), sn=sn
        )
        self.downsample_in_noise = Residual3DConvBlock(
            _facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.combined1 = GeneratorCombined1Block(
            _facify(256, fac), _facify(128, fac), sn=sn
        )
        self.combined2 = GeneratorCombined2Block(_facify(256, fac), sn=sn)

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(384, fac), _facify(256, fac), 3, 1, trans=True, sn=sn
            ),
            Residual3DConvBlock(_facify(256, fac), _facify(256, fac), 3, 1, sn=sn),
            specnorm(nn.Conv3d(_facify(256, fac), 53, kernel_size=1, stride=1)),
            nn.Sigmoid() if not testmode else nn.Tanh(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)
        self.testmode = testmode

    def forward(self, z, c):
        # c is torch.Size([64, 63, 16, 16, 16])
        embedded_c = self.embed_condition(c)  # torch.Size([64, 16, 16, 16, 16])

        embed_z = self.embed_noise_in(z)
        embedded_c = self.combine_in_noise(embed_z, embedded_c)
        embedded_c = self.downsample_in_noise(embedded_c)

        down1 = self.downsample_cond1(embedded_c)  # torch.Size([64, 16, 8, 8, 8])
        down2 = self.downsample_cond2(down1)  # torch.Size([64, 16, 4, 4, 4])

        embedded_z = self.embed_noise(z)  # torch.Size([64, 16, 4, 4, 4])

        out = self.combined1(embedded_z, down2)  # torch.Size([64, 16, 8, 8, 8])

        c_down1 = torch.cat((out, down1), dim=1)  # torch.Size([64, 32, 8, 8, 8])
        out = self.combined2(c_down1)  # torch.Size([64, 32, 16, 16, 16])

        out = torch.cat((out, embedded_c), dim=1)  # torch.Size([64, 48, 16, 16, 16])
        out = (
            c[:, :53, ...] + self.to_image(out) if self.testmode else self.to_image(out)
        )  # torch.Size([64, 53, 16, 16, 16])

        return out, embedded_z


class Decoder2(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac=1,
        testmode=False,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(128, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(128, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )

        self.embed_noise_in = EmbedNoise(z_dim, 1, dim=16, sn=sn)

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(128, fac) + 1, _facify(256, fac), 3, 1, trans=True, sn=sn
            ),
            Residual3DConvBlock(_facify(256, fac), _facify(256, fac), 3, 1, sn=sn),
            specnorm(nn.Conv3d(_facify(256, fac), 53, kernel_size=1, stride=1)),
            nn.Sigmoid() if not testmode else nn.Tanh(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)
        self.testmode = testmode

    def forward(self, z, c):
        # c is torch.Size([64, 63, 16, 16, 16])
        embedded_c = self.embed_condition(c)

        embed_z = self.embed_noise_in(z)
        combined_input = torch.cat((embedded_c, embed_z), dim=1)
        out = self.to_image(combined_input)

        return out


class Decoder3(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        out_channels,
        resolution,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(512, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(512, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(512, fac), _facify(512, fac), kernel_size=3, stride=1, sn=sn,
            ),
            Residual3DConvBlock(
                _facify(512, fac), _facify(512, fac), kernel_size=3, stride=1, sn=sn,
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )

        self.embed_noise_in = EmbedNoise(z_dim, 1, dim=resolution, sn=sn)

        to_image_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=_facify(512, fac) + 1,
                    out_channels=_facify(512, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(512, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(512, fac), _facify(512, fac), kernel_size=3, stride=1, sn=sn,
            ),
            Residual3DConvBlock(
                _facify(512, fac), _facify(512, fac), kernel_size=3, stride=1, sn=sn,
            ),
            specnorm(
                nn.Conv3d(
                    in_channels=_facify(512, fac),
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=compute_same_padding(1, 1, 1),
                )
            ),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

        # self.attend = QubicAttention_3d(_facify(256, fac) + 1)
        # self.attend = Attention_block(1, _facify(256, fac), _facify(256, fac))

    def forward(self, z, c):
        # c is torch.Size([64, 63, 16, 16, 16])
        embedded_c = self.embed_condition(c)

        embed_z = self.embed_noise_in(z)
        combined_input = torch.cat((embedded_c, embed_z), dim=1)
        # combined_input = self.attend(embed_z, embedded_c)

        # attned
        # combined_input = self.attend(combined_input)
        # combined_input = self.attend(combined_input)

        out = self.to_image(combined_input)

        return out


class QubicAttention_3d(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(QubicAttention_3d, self).__init__()

        self.q_conv_t = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.k_conv_t = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.v_cov_t = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.q_conv_h = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.k_conv_h = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.v_conv_h = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )

        self.q_conv_v = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.k_conv_v = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.v_conv_v = nn.Conv3d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )

        self.conv_s = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q_t = self.q_conv_t(x)
        k_t = self.k_conv_t(x)
        v_t = self.v_cov_t(x)

        q_h = self.q_conv_h(x)
        k_h = self.k_conv_h(x)
        v_h = self.v_conv_h(x)

        q_v = self.q_conv_v(x)
        k_v = self.k_conv_v(x)
        v_v = self.v_conv_v(x)

        short = self.conv_s(x)

        #  n*25*2048*7*7  reshape to n*25*7*7*2048
        #  n*2048*25*7*7  reshape to n*25*7*7*2048
        q_t = q_t.permute(0, 2, 3, 4, 1)
        k_t = k_t.permute(0, 2, 3, 4, 1)
        v_t = v_t.permute(0, 2, 3, 4, 1)
        q_h = q_h.permute(0, 2, 3, 4, 1)
        k_h = k_h.permute(0, 2, 3, 4, 1)
        v_h = v_h.permute(0, 2, 3, 4, 1)
        q_v = q_v.permute(0, 2, 3, 4, 1)
        k_v = k_v.permute(0, 2, 3, 4, 1)
        v_v = v_v.permute(0, 2, 3, 4, 1)

        # temporal attention
        # {n*7*7}*25*2048
        #  x shape:: n*2048*25*7*7
        q_t = q_t.contiguous().view(
            (x.shape[0] * x.shape[3] * x.shape[4]), x.shape[2], -1
        )
        k_t = k_t.contiguous().view(
            (x.shape[0] * x.shape[3] * x.shape[4]), x.shape[2], -1
        )
        v_t = v_t.contiguous().view(
            (x.shape[0] * x.shape[3] * x.shape[4]), x.shape[2], -1
        )

        q_t = q_t.permute(0, 2, 1)  # {n*7*7}*2048*25
        attention_t = F.softmax(k_t @ q_t, -1)  # {n*7*7}*25*25
        out_t = attention_t @ v_t  # {n*7*7}*25*2048

        out_t = out_t.contiguous().view(
            x.shape[0], x.shape[3], x.shape[4], x.shape[2], x.shape[1]
        )  # n*7*7*25*2048
        out_t = out_t.permute(0, 4, 3, 1, 2)  # n*25*2048*7*7

        # horizontal attention
        #  x shape:: n*2048*25*7*7   q_h shape:: n*25*7*7*2048
        # target shape:: {n*25*7}*7*2048
        q_h = q_h.contiguous().view(
            (x.shape[0] * x.shape[2] * x.shape[4]), x.shape[3], -1
        )
        k_h = k_h.contiguous().view(
            (x.shape[0] * x.shape[2] * x.shape[4]), x.shape[3], -1
        )
        v_h = v_h.contiguous().view(
            (x.shape[0] * x.shape[2] * x.shape[4]), x.shape[3], -1
        )

        q_h = q_h.permute(0, 2, 1)  # {n*25*7}*2048*7
        attention_h = F.softmax(k_h @ q_h, -1)  # {n*25*7}*7*7
        out_h = attention_h @ v_h  # {n*25*7}*7*2048

        out_h = out_h.contiguous().view(
            x.shape[0], x.shape[2], x.shape[4], x.shape[3], x.shape[1]
        )  # n*25*7*7*2048
        out_h = out_h.permute(0, 4, 1, 3, 2)  # n*25*2048*7*7

        # vertical attention
        #  x shape:: n*2048*25*7*7   q shape:: n*25*7*7*2048
        # target shape:: {n*25*7}*7*2048
        q_v = q_v.contiguous().view(
            (x.shape[0] * x.shape[2] * x.shape[3]), x.shape[4], -1
        )
        k_v = k_v.contiguous().view(
            (x.shape[0] * x.shape[2] * x.shape[3]), x.shape[4], -1
        )
        v_v = v_v.contiguous().view(
            (x.shape[0] * x.shape[2] * x.shape[3]), x.shape[4], -1
        )

        q_v = q_v.permute(0, 2, 1)  # {n*25*7}*2048*7
        attention_v = F.softmax(k_v @ q_v, -1)  # {n*25*7}*7*7
        out_v = attention_v @ v_v  # {n*25*7}*7*2048

        out_v = out_v.contiguous().view(
            x.shape[0], x.shape[2], x.shape[3], x.shape[4], x.shape[1]
        )  # n*25*7*7*2048
        out_v = out_v.permute(0, 4, 1, 2, 3)  # n*25*2048*7*7

        out = out_t + out_h + out_v + short  # n*25*2048*7*7
        return out


class RCCAModule_3d(nn.Module):
    def __init__(self, in_channels, recurrence=2):
        super(RCCAModule_3d, self).__init__()
        self.recurrence = recurrence
        self.cca = QubicAttention_3d(in_channels)

    def forward(self, x):
        output = x
        for i in range(self.recurrence):
            output = self.cca(output)
        output_mean = torch.mean(output, dim=[2, 3, -1])
        return output_mean


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class Decoder4(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac,
        out_channels,
        resolution,
        testmode=False,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            # specnorm(
            #    nn.Conv3d(
            #        in_channels=condition_n_channels,
            #        out_channels=_facify(256, fac),
            #        kernel_size=3,
            #        stride=1,
            #        padding=compute_same_padding(3, 1, 1),
            #    )
            # ),
            # nn.GroupNorm(1, num_channels=_facify(256, fac)),
            # nn.LeakyReLU(),
            Residual3DConvBlock(
                condition_n_channels,
                _facify(256, fac),
                kernel_size=3,
                stride=1,
                trans=True,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac), _facify(256, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(256, fac), _facify(256, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond1 = Residual3DConvBlock(
            _facify(256, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )
        self.downsample_cond2 = Residual3DConvBlock(
            _facify(256, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        embed_noise_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=z_dim // 8,
                    out_channels=_facify(256, fac),
                    kernel_size=1,
                    stride=1,
                    padding=compute_same_padding(1, 1, 1),
                )
            ),
            nn.LeakyReLU(),
        ]
        self.embed_noise = nn.Sequential(*tuple(embed_noise_blocks)).to(device=device)

        self.combined1 = GeneratorCombined1Block(
            _facify(512, fac), _facify(256, fac), sn=sn
        )
        self.combined2 = GeneratorCombined2Block(_facify(512, fac), sn=sn)

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(512, fac) + _facify(256, fac),
                _facify(512, fac),
                kernel_size=3,
                stride=1,
                trans=True,
                sn=sn,
            ),
            Residual3DConvBlock(_facify(512, fac), _facify(512, fac), 3, 1, sn=sn),
            Residual3DConvBlock(_facify(512, fac), _facify(512, fac), 3, 1, sn=sn),
            Residual3DConvBlock(_facify(512, fac), _facify(512, fac), 3, 1, sn=sn),
            specnorm(
                nn.Conv3d(_facify(512, fac), out_channels, kernel_size=1, stride=1)
            ),
            nn.Sigmoid() if not testmode else nn.Tanh(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)
        self.testmode = testmode

    def forward(self, z, c):
        # c is torch.Size([64, 63, 16, 16, 16])
        embedded_c = self.embed_condition(c)  # torch.Size([64, 16, 16, 16, 16])

        # embedded_c = self.combine_in_noise(embed_z, embedded_c)
        # embedded_c = self.downsample_in_noise(embedded_c)

        down1 = self.downsample_cond1(embedded_c)  # torch.Size([64, 16, 8, 8, 8])
        down2 = self.downsample_cond2(down1)  # torch.Size([64, 16, 4, 4, 4])

        # embedded_z = self.embed_noise(z)  # torch.Size([64, 16, 4, 4, 4])
        embedded_z = z.view(embedded_c.size(0), z.size(1) // 8, 2, 2, 2)
        embedded_z = self.embed_noise(embedded_z)

        out = self.combined1(embedded_z, down2)  # torch.Size([64, 16, 8, 8, 8])

        c_down1 = torch.cat((out, down1), dim=1)  # torch.Size([64, 32, 8, 8, 8])
        out = self.combined2(c_down1)  # torch.Size([64, 32, 16, 16, 16])

        out = torch.cat((out, embedded_c), dim=1)  # torch.Size([64, 48, 16, 16, 16])
        out = self.to_image(out)  # torch.Size([64, 53, 16, 16, 16])

        return out


class Decoder5(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac,
        out_channels,
        resolution,
        testmode=False,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            # specnorm(
            #    nn.Conv3d(
            #        in_channels=condition_n_channels,
            #        out_channels=_facify(256, fac),
            #        kernel_size=3,
            #        stride=1,
            #        padding=compute_same_padding(3, 1, 1),
            #    )
            # ),
            # nn.GroupNorm(1, num_channels=_facify(256, fac)),
            # nn.LeakyReLU(),
            Residual3DConvBlock(
                condition_n_channels,
                _facify(256, fac),
                kernel_size=3,
                stride=1,
                trans=True,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac), _facify(256, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(256, fac), _facify(256, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(256, fac), _facify(256, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(256, fac), _facify(256, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )

        combine_noise1 = [
            # specnorm(
            #    nn.Conv3d(
            #        in_channels=_facify(256, fac) + z_dim // 8,
            #        out_channels=_facify(256, fac),
            #        kernel_size=1,
            #        stride=1,
            #        padding=compute_same_padding(1, 1, 1),
            #    )),
            # nn.GroupNorm(1, num_channels=_facify(256, fac)),
            # nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(256, fac) + z_dim // 8,
                _facify(256, fac),
                kernel_size=3,
                stride=1,
                trans=True,
                sn=sn,
            ),
        ]
        self.combine_noise1 = nn.Sequential(*tuple(combine_noise1)).to(device=device)
        self.pad1 = nn.ReplicationPad3d(3)

        downsample_cond1_blocks = [
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
        ]
        self.downsample_cond1 = nn.Sequential(*tuple(downsample_cond1_blocks)).to(
            device=device
        )

        combine_noise2 = [
            # specnorm(
            #    nn.Conv3d(
            #        in_channels=_facify(256, fac) + z_dim // 8,
            #        out_channels=_facify(256, fac),
            #        kernel_size=1,
            #        stride=1,
            #        padding=compute_same_padding(1, 1, 1),
            #    )),
            # nn.GroupNorm(1, num_channels=_facify(256, fac)),
            # nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(256, fac) + z_dim // 8,
                _facify(256, fac),
                kernel_size=3,
                stride=1,
                trans=True,
                sn=sn,
            ),
        ]
        self.combine_noise2 = nn.Sequential(*tuple(combine_noise2)).to(device=device)
        self.pad2 = nn.ReplicationPad3d(1)

        downsample_cond2_blocks = [
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
        ]
        self.downsample_cond2 = nn.Sequential(*tuple(downsample_cond2_blocks)).to(
            device=device
        )

        embed_noise_blocks = [
            # specnorm(
            #    nn.Conv3d(
            #        in_channels=z_dim // 8,
            #        out_channels=_facify(256, fac),
            #        kernel_size=1,
            #        stride=1,
            #        padding=compute_same_padding(1, 1, 1),
            #    )),
            # nn.LeakyReLU(),
            Residual3DConvBlock(
                z_dim // 8,
                _facify(256, fac),
                kernel_size=3,
                stride=1,
                trans=True,
                sn=sn,
            ),
        ]
        self.embed_noise = nn.Sequential(*tuple(embed_noise_blocks)).to(device=device)

        self.combined1 = GeneratorCombined1Block(
            _facify(512, fac), _facify(256, fac), sn=sn
        )
        self.combined2 = GeneratorCombined2Block(_facify(512, fac), sn=sn)

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(512, fac) + _facify(256, fac),
                _facify(512, fac),
                kernel_size=3,
                stride=1,
                trans=True,
                sn=sn,
            ),
            Residual3DConvBlock(_facify(512, fac), _facify(512, fac), 3, 1, sn=sn),
            specnorm(
                nn.Conv3d(_facify(512, fac), out_channels, kernel_size=1, stride=1)
            ),
            nn.Sigmoid() if not testmode else nn.Tanh(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)
        self.testmode = testmode

    def forward(self, z, c):
        # c is torch.Size([64, 63, 16, 16, 16])
        embedded_c = self.embed_condition(c)  # torch.Size([64, 16, 16, 16, 16])

        # embedded_c = self.combine_in_noise(embed_z, embedded_c)
        # embedded_c = self.downsample_in_noise(embedded_c)

        embedded_z = z.view(embedded_c.size(0), z.size(1) // 8, 2, 2, 2)

        together = self.combine_noise1(
            torch.cat((embedded_c, self.pad1(embedded_z)), dim=1)
        )
        down1 = self.downsample_cond1(together)  # torch.Size([64, 16, 8, 8, 8])

        together = self.combine_noise2(torch.cat((down1, self.pad2(embedded_z)), dim=1))
        down2 = self.downsample_cond2(together)  # torch.Size([64, 16, 4, 4, 4])

        # embedded_z = self.embed_noise(z)  # torch.Size([64, 16, 4, 4, 4])
        embedded_z = self.embed_noise(embedded_z)

        out = self.combined1(embedded_z, down2)  # torch.Size([64, 16, 8, 8, 8])

        c_down1 = torch.cat((out, down1), dim=1)  # torch.Size([64, 32, 8, 8, 8])
        out = self.combined2(c_down1)  # torch.Size([64, 32, 16, 16, 16])

        out = torch.cat((out, embedded_c), dim=1)  # torch.Size([64, 48, 16, 16, 16])
        out = self.to_image(out)  # torch.Size([64, 53, 16, 16, 16])

        return out


class Encoder4(nn.Module):
    def __init__(
        self,
        width,
        height,
        depth,
        in_channels,
        latent_dim,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)

        encoder_modules = [
            # specnorm(
            #    nn.Conv3d(
            #        in_channels=in_channels,
            #        out_channels=_facify(512, fac),
            #        kernel_size=3,
            #        stride=1,
            #        padding=compute_same_padding(3, 1, 1),
            #    )
            # ),
            # nn.GroupNorm(1, _facify(512, fac)),
            # nn.LeakyReLU(),
            Residual3DConvBlock(
                in_channels,
                _facify(512, fac),
                kernel_size=5,
                stride=1,
                trans=True,
                sn=sn,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            nn.Flatten(),
        ]
        self.featurizer = nn.Sequential(*encoder_modules)
        num_features = _facify(512, fac)
        # num_features = width * height * depth
        # num_features = (width // 4) * (height // 4) * (depth // 4) * _facify(512, fac)
        # num_features = (width // 8) * (height // 8) * (depth // 8) * _facify(1024, fac)
        self.to_latent_mu = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )
        self.to_latent_logvar = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )

    def forward(self, inputs):
        features = self.featurizer(inputs)
        mu = self.to_latent_mu(features)
        logvar = self.to_latent_logvar(features)
        return mu, logvar


class Decoder6(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac,
        out_channels,
        resolution,
        testmode=False,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            # Residual3DConvBlock(
            #    condition_n_channels,
            #    _facify(256, fac),
            #    kernel_size=3,
            #    stride=1,
            #    trans=True,
            #    sn=sn,
            # ),
            specnorm(
                nn.Conv3d(
                    condition_n_channels,
                    _facify(256, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, _facify(256, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(256, fac), _facify(256, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(256, fac), _facify(256, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks))

        combine_noise1 = [
            # Residual3DConvBlock(
            #    _facify(256, fac) + z_dim // 8,
            #    _facify(256, fac),
            #    kernel_size=3,
            #    stride=1,
            #    trans=True,
            #    sn=sn,
            # ),
            specnorm(
                nn.Conv3d(
                    _facify(256, fac) + z_dim // 8,
                    _facify(256, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            # nn.GroupNorm(1, _facify(256, fac)),
            nn.LeakyReLU(),
        ]
        self.combine_noise1 = nn.Sequential(*tuple(combine_noise1))
        self.pad1 = nn.ReplicationPad3d(5)

        downsample_cond1_blocks = [
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
        ]
        self.downsample_cond1 = nn.Sequential(*tuple(downsample_cond1_blocks))

        combine_noise2 = [
            # Residual3DConvBlock(
            #    _facify(256, fac) + z_dim // 8,
            #    _facify(256, fac),
            #    kernel_size=3,
            #    stride=1,
            #    trans=True,
            #    sn=sn,
            # ),
            specnorm(
                nn.Conv3d(
                    _facify(256, fac) + z_dim // 8,
                    _facify(256, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            # nn.GroupNorm(1, _facify(256, fac)),
            nn.LeakyReLU(),
        ]
        self.combine_noise2 = nn.Sequential(*tuple(combine_noise2))
        self.pad2 = nn.ReplicationPad3d(2)

        downsample_cond2_blocks = [
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
            Residual3DConvBlock(
                _facify(256, fac),
                n_filters=_facify(256, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
            ),
        ]
        self.downsample_cond2 = nn.Sequential(*tuple(downsample_cond2_blocks))

        self.pad3 = nn.ReplicationPad3d((0, 1, 0, 1, 0, 1))
        embed_noise_blocks = [
            # Residual3DConvBlock(
            #    z_dim // 8,
            #    _facify(256, fac),
            #    kernel_size=3,
            #    stride=1,
            #    trans=True,
            #    sn=sn,
            # ),
            specnorm(
                nn.Conv3d(
                    z_dim // 8,
                    _facify(256, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            # nn.GroupNorm(1, _facify(256, fac)),
            nn.LeakyReLU(),
        ]
        self.embed_noise = nn.Sequential(*tuple(embed_noise_blocks))

        self.combined1 = GeneratorCombined1Block(
            _facify(512, fac), _facify(256, fac), sn=sn
        )
        self.combined2 = GeneratorCombined2Block(_facify(512, fac), sn=sn)

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(512, fac) + _facify(256, fac),
                _facify(512, fac),
                kernel_size=3,
                stride=1,
                trans=True,
                sn=sn,
            ),
            Residual3DConvBlock(_facify(512, fac), _facify(512, fac), 3, 1, sn=sn),
            Residual3DConvBlock(_facify(512, fac), _facify(512, fac), 3, 1, sn=sn),
            specnorm(
                nn.Conv3d(_facify(512, fac), out_channels, kernel_size=1, stride=1)
            ),
            nn.Sigmoid() if not testmode else nn.Tanh(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks))
        self.testmode = testmode

    def forward(self, z, c):
        # c is torch.Size([64, 63, 16, 16, 16])
        embedded_c = self.embed_condition(c)  # torch.Size([64, 16, 16, 16, 16])

        # embedded_c = self.combine_in_noise(embed_z, embedded_c)
        # embedded_c = self.downsample_in_noise(embedded_c)

        embedded_z = z.view(embedded_c.size(0), z.size(1) // 8, 2, 2, 2)

        together = self.combine_noise1(
            torch.cat((embedded_c, self.pad1(embedded_z)), dim=1)
        )
        down1 = self.downsample_cond1(together)  # torch.Size([64, 16, 8, 8, 8])

        together = self.combine_noise2(torch.cat((down1, self.pad2(embedded_z)), dim=1))
        down2 = self.downsample_cond2(together)  # torch.Size([64, 16, 4, 4, 4])

        # embedded_z = self.embed_noise(z)  # torch.Size([64, 16, 4, 4, 4])
        embedded_z = self.embed_noise(self.pad3(embedded_z))

        out = self.combined1(embedded_z, down2)  # torch.Size([64, 16, 8, 8, 8])

        c_down1 = torch.cat((out, down1), dim=1)  # torch.Size([64, 32, 8, 8, 8])
        out = self.combined2(c_down1)  # torch.Size([64, 32, 16, 16, 16])

        out = torch.cat((out, embedded_c), dim=1)  # torch.Size([64, 48, 16, 16, 16])
        out = self.to_image(out)  # torch.Size([64, 53, 16, 16, 16])

        return out


class Encoder6(nn.Module):
    def __init__(
        self,
        width,
        height,
        depth,
        in_channels,
        latent_dim,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)

        encoder_modules = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(512, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, _facify(512, fac)),
            nn.LeakyReLU(),
            # Residual3DConvBlock(
            #    in_channels,
            #    _facify(512, fac),
            #    kernel_size=3,
            #    stride=1,
            #    trans=True,
            #    sn=sn,
            # ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            # nn.AvgPool3d(2),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            # nn.AvgPool3d(2),
            # Residual3DConvBlock(
            #    in_channels=_facify(512, fac),
            #    n_filters=_facify(512, fac),
            #    kernel_size=3,
            #    stride=2,
            #    sn=sn,
            #    device=device,
            # ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            # Residual3DConvBlock(
            #    in_channels=_facify(512, fac),
            #    n_filters=_facify(512, fac),
            #    kernel_size=3,
            #    stride=2,
            #    sn=sn,
            #    device=device,
            # ),
            # nn.AvgPool3d(2),
            # Residual3DConvBlock(
            #    in_channels=_facify(512, fac),
            #    n_filters=_facify(512, fac),
            #    kernel_size=3,
            #    stride=1,
            #    sn=sn,
            #    device=device,
            # ),
            nn.Flatten(),
        ]
        self.featurizer = nn.Sequential(*encoder_modules)
        num_features = _facify(512, fac) * 27
        # num_features = width * height * depth
        # num_features = (width // 4) * (height // 4) * (depth // 4) * _facify(512, fac)
        # num_features = (width // 8) * (height // 8) * (depth // 8) * _facify(1024, fac)
        self.to_latent_mu = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )
        self.to_latent_logvar = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )

    def forward(self, inputs):
        features = self.featurizer(inputs)
        mu = self.to_latent_mu(features)
        logvar = self.to_latent_logvar(features)
        return mu, logvar


class Decoder7(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        out_channels,
        resolution,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(512, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(512, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(512, fac), _facify(512, fac), kernel_size=3, stride=1, sn=sn,
            ),
            Residual3DConvBlock(
                _facify(512, fac), _facify(512, fac), kernel_size=3, stride=1, sn=sn,
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )

        self.pad = nn.ReplicationPad3d(5)
        embed_noise_blocks = [
            specnorm(
                nn.Conv3d(
                    z_dim // 8,
                    _facify(512, fac),
                    kernel_size=5,
                    stride=1,
                    padding=compute_same_padding(5, 1, 1),
                )
            ),
            # nn.GroupNorm(1, _facify(256, fac)),
            nn.LeakyReLU(),
        ]
        self.embed_noise = nn.Sequential(*tuple(embed_noise_blocks))

        to_image_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=_facify(512, fac) + _facify(512, fac),
                    out_channels=_facify(1024, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(1024, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(1024, fac), _facify(1024, fac), kernel_size=3, stride=1, sn=sn,
            ),
            Residual3DConvBlock(
                _facify(1024, fac), _facify(1024, fac), kernel_size=3, stride=1, sn=sn,
            ),
            specnorm(
                nn.Conv3d(
                    in_channels=_facify(1024, fac),
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=compute_same_padding(1, 1, 1),
                )
            ),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

        # self.attend = QubicAttention_3d(_facify(256, fac) + 1)
        # self.attend = Attention_block(1, _facify(256, fac), _facify(256, fac))

    def forward(self, z, c):
        # c is torch.Size([64, 63, 16, 16, 16])
        embedded_c = self.embed_condition(c)

        embedded_z = z.view(embedded_c.size(0), z.size(1) // 8, 2, 2, 2)
        embedded_z = self.embed_noise(self.pad(embedded_z))

        combined_input = torch.cat((embedded_c, embedded_z), dim=1)

        out = self.to_image(combined_input)

        return out


class Encoder8(nn.Module):
    def __init__(
        self,
        width,
        height,
        depth,
        in_channels,
        latent_dim,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)

        encoder_modules = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(512, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, _facify(512, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(1024, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
                trans=True,
            ),
            Residual3DConvBlock(
                in_channels=_facify(1024, fac),
                n_filters=_facify(1024, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(1024, fac),
                n_filters=_facify(1024, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(1024, fac),
                n_filters=_facify(1024, fac),
                kernel_size=3,
                stride=2,
                sn=sn,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(1024, fac),
                n_filters=_facify(1024, fac),
                kernel_size=3,
                stride=1,
                sn=sn,
                device=device,
            ),
            nn.Flatten(),
        ]
        # encoder_modules = [
        #    specnorm(
        #        nn.Conv3d(
        #            in_channels=in_channels,
        #            out_channels=_facify(512, fac),
        #            kernel_size=3,
        #            stride=1,
        #            padding=compute_same_padding(3, 1, 1),
        #        )),
        #    nn.GroupNorm(1, _facify(512, fac)),
        #    nn.LeakyReLU(),
        #    Residual3DConvBlock(
        #        in_channels=_facify(512, fac),
        #        n_filters=_facify(512, fac),
        #        kernel_size=3,
        #        stride=1,
        #        sn=sn,
        #        device=device,
        #    ),
        #    Residual3DConvBlock(
        #        in_channels=_facify(512, fac),
        #        n_filters=_facify(512, fac),
        #        kernel_size=3,
        #        stride=2,
        #        sn=sn,
        #        device=device,
        #    ),
        #    Residual3DConvBlock(
        #        in_channels=_facify(512, fac),
        #        n_filters=_facify(512, fac),
        #        kernel_size=3,
        #        stride=1,
        #        sn=sn,
        #        device=device,
        #    ),
        #    Residual3DConvBlock(
        #        in_channels=_facify(512, fac),
        #        n_filters=_facify(512, fac),
        #        kernel_size=3,
        #        stride=2,
        #        sn=sn,
        #        device=device,
        #    ),
        #    Residual3DConvBlock(
        #        in_channels=_facify(512, fac),
        #        n_filters=_facify(512, fac),
        #        kernel_size=3,
        #        stride=1,
        #        sn=sn,
        #        device=device,
        #    ),
        #    Residual3DConvBlock(
        #        in_channels=_facify(512, fac),
        #        n_filters=_facify(512, fac),
        #        kernel_size=3,
        #        stride=1,
        #        sn=sn,
        #        device=device,
        #    ),
        #    nn.Flatten(),
        # ]
        # num_features = _facify(512, fac) * 27
        num_features = _facify(1024, fac) * 27
        self.featurizer = nn.Sequential(*encoder_modules)
        # num_features = width * height * depth
        # num_features = (width // 4) * (height // 4) * (depth // 4) * _facify(512, fac)
        # num_features = (width // 8) * (height // 8) * (depth // 8) * _facify(1024, fac)
        self.to_latent_mu = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )
        self.to_latent_logvar = specnorm(
            nn.Linear(in_features=num_features, out_features=latent_dim,)
        )

    def forward(self, inputs):
        features = self.featurizer(inputs)
        mu = self.to_latent_mu(features)
        logvar = self.to_latent_logvar(features)
        return mu, logvar
