import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math
from einops.layers.torch import Rearrange
from einops import rearrange


class CausalConv1d(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        dilation
    ) -> None:
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class WaveNetBlock(nn.Module):
    def __init__(
        self, 
        dilation_rate,
        input_channels,
        skip_channels = 128,
        residual_channels = 64,
        dilation_channels = 32,
        kernel_size = 3,
        dropout_rate = 0.2
    ) -> None:
        super(WaveNetBlock, self).__init__()
        self.dilated_conv = CausalConv1d(input_channels, dilation_channels, kernel_size, dilation=dilation_rate)
        self.batch_norm = nn.BatchNorm1d(dilation_channels)
        self.residual_conv = nn.Conv1d(dilation_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.residual_adjust = nn.Conv1d(residual_channels, input_channels, 1)

    def forward(self, x, skip_sum):
        x_in = x
        x = self.dilated_conv(x)
        x = self.batch_norm(x)
        tanh_out = self.tanh(x)
        sigmoid_out = self.sigmoid(x)
        x = tanh_out * sigmoid_out
        x = self.dropout(x)
        skip_out = self.skip_conv(x)
        skip_sum = skip_sum + skip_out if skip_sum is not None else skip_out
        x = self.residual_conv(x)
        x = self.residual_adjust(x)
        x = x + x_in
        return x, skip_sum

class WaveNet(nn.Module):
    def __init__(
        self, 
        dilation_rates,
        input_channels,
        num_classes,
        skip_channels = 128
    ) -> None:
        super(WaveNet, self).__init__()
        self.blocks = nn.ModuleList([WaveNetBlock(d, input_channels) for d in dilation_rates])
        self.relu = nn.ReLU()
        self.conv1x1_out1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.conv1x1_out2 = nn.Conv1d(skip_channels, num_classes, 1)

    def forward(self, x, subject_idxs):
        subject_idxs = torch.reshape(subject_idxs, (len(subject_idxs),1,1))
        subject_idx = subject_idxs
        for i in range(270):
            subject_idx = torch.cat([subject_idx, subject_idxs], dim=1) 
        x = torch.cat([x, subject_idx], dim=2)
        skip_sum = None
        for block in self.blocks:
            x, skip_sum = block(x, skip_sum)
        x = self.relu(skip_sum)
        x = self.relu(self.conv1x1_out1(x))
        x = self.conv1x1_out2(x)
        x = x.mean(dim=2)
        return x



class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        subject_idxs = torch.reshape(subject_idxs, (len(subject_idxs),1,1))
        subject_idx = subject_idxs
        for i in range(270):
            subject_idx = torch.cat([subject_idx, subject_idxs], dim=1) 
        X = torch.cat([X, subject_idx], dim=2)
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        # self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X) + X
        # X = F.gelu(self.batchnorm2(X))

        return self.dropout(X)
