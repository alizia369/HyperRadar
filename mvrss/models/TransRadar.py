import torch
import torch.nn as nn
import torch.nn.functional as F
from .hyperradar_blocks import LearnableHypergraph, CrossViewUOT, ChannelGate


class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x




class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view.
    Same implementation as the original MVRSS paper.

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type, k_size = 3):
        super().__init__()
        self.signal_type = signal_type
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=128, k_size=(k_size,3,3),
                                                      pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                  pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)
            x1_down = self.doppler_max_pool(x1_pad)
        else:
            x1_down = self.max_pool(x1)

        x2 = self.double_conv_block2(x1_down)
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2)

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x3





class TransRad(nn.Module):
    """HyperRadar adapted to the *original TransRadar* training interface.

    Call: TransRad(n_classes, n_frames, deform_k, depth, channels)
    Forward: (rd, ra, ad) -> (rd_logits, ra_logits)

    Three unmodified TransRadar encoders feed independent projections and
    per-view learnable hypergraphs, followed by 4 cross-view UOT transfers,
    separate cascaded channel gates and the original two decoder branches.

    All optional hyperparameters are constructor defaults because official
    train.py / test.py pass only the legacy 5 arguments. To modify them, edit
    the constants here or add optional arguments to those entry points.
    """
    def __init__(self, n_classes, n_frames,
                 deform_k=(3,3,3,3,3,3,3,3), depth=8, channels=64,
                 hyperedges=8, epsilon=0.05, tau=0.8, sinkhorn_iterations=30,
                 transport_grid=16):
        super().__init__()
        self.n_classes=n_classes
        self.n_frames=n_frames
        if n_frames!=5:
            raise ValueError('Original TransRadar temporal encoder requires 5 frames.')
        k_size=n_frames//2+1
        self.rd_encoding_branch=EncodingBranch('range_doppler',k_size=k_size)
        self.ra_encoding_branch=EncodingBranch('range_angle',k_size=k_size)
        self.ad_encoding_branch=EncodingBranch('angle_doppler',k_size=k_size)

        # Equation 1: independent latent view embeddings at 64x64 (CARRADA).
        self.rd_project=ConvBlock(128,channels,1,0,1)
        self.ra_project=ConvBlock(128,channels,1,0,1)
        self.ad_project=ConvBlock(128,channels,1,0,1)

        # Equation 2: one learnable hypergraph per radar view.
        self.rd_hypergraph=LearnableHypergraph(channels,hyperedges)
        self.ra_hypergraph=LearnableHypergraph(channels,hyperedges)
        self.ad_hypergraph=LearnableHypergraph(channels,hyperedges)

        # Equations 3-5: RD receives RA and AD, RA receives RD and AD.
        self.crossview_uot=CrossViewUOT(channels,transport_grid,
                                        epsilon,tau,sinkhorn_iterations)

        # Equation 6: paper uses cascaded adaptive channel gating.
        # The legacy TransRadar deform_k argument is intentionally unused.
        self.rd_attention=nn.Sequential(*[ChannelGate(channels) for _ in range(depth)])
        self.ra_attention=nn.Sequential(*[ChannelGate(channels) for _ in range(depth)])

        # Equation 7: preserve the original, distinct RD and RA decoders.
        self.rd_single_conv_block2_1x1=ConvBlock(channels,128,1,0,1)
        self.ra_single_conv_block2_1x1=ConvBlock(channels,128,1,0,1)
        self.rd_upconv1=nn.ConvTranspose2d(128,128,(2,1),stride=(2,1))
        self.ra_upconv1=nn.ConvTranspose2d(128,128,2,stride=2)
        self.rd_double_conv_block1=DoubleConvBlock(128,128,3,1,1)
        self.ra_double_conv_block1=DoubleConvBlock(128,128,3,1,1)
        self.rd_upconv2=nn.ConvTranspose2d(128,128,(2,1),stride=(2,1))
        self.ra_upconv2=nn.ConvTranspose2d(128,128,2,stride=2)
        self.rd_double_conv_block2=DoubleConvBlock(128,128,3,1,1)
        self.ra_double_conv_block2=DoubleConvBlock(128,128,3,1,1)
        self.rd_final=nn.Conv2d(128,n_classes,1)
        self.ra_final=nn.Conv2d(128,n_classes,1)

    def forward(self,x_rd,x_ra,x_ad):
        rd=self.rd_project(self.rd_encoding_branch(x_rd))
        ra=self.ra_project(self.ra_encoding_branch(x_ra))
        ad=self.ad_project(self.ad_encoding_branch(x_ad))
        rd=self.rd_hypergraph(rd)
        ra=self.ra_hypergraph(ra)
        ad=self.ad_hypergraph(ad)
        rd,ra=self.crossview_uot(rd,ra,ad)
        rd=self.rd_attention(rd)
        ra=self.ra_attention(ra)
        rd=self.rd_single_conv_block2_1x1(rd)
        ra=self.ra_single_conv_block2_1x1(ra)
        rd=self.rd_double_conv_block2(self.rd_upconv2(self.rd_double_conv_block1(self.rd_upconv1(rd))))
        ra=self.ra_double_conv_block2(self.ra_upconv2(self.ra_double_conv_block1(self.ra_upconv1(ra))))
        return self.rd_final(rd),self.ra_final(ra)
