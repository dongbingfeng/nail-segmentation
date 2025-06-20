"""
Model Components for U-Net Architecture

Provides reusable building blocks for the U-Net model including
encoder blocks, decoder blocks, and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """
    Basic convolutional block with optional batch normalization and dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        batch_norm: bool = True,
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        """
        Initialize convolutional block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            stride: Stride size
            batch_norm: Whether to use batch normalization
            dropout: Dropout probability
            activation: Activation function ('relu', 'leaky_relu', 'gelu')
        """
        super(ConvBlock, self).__init__()
        
        layers = []
        
        # Convolution
        layers.append(nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=not batch_norm
        ))
        
        # Batch normalization
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder block for U-Net with two convolutional layers
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        batch_norm: bool = True
    ):
        """
        Initialize encoder block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super(EncoderBlock, self).__init__()
        
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            batch_norm=batch_norm,
            dropout=dropout
        )
        
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            batch_norm=batch_norm,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net with two convolutional layers
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        batch_norm: bool = True
    ):
        """
        Initialize decoder block
        
        Args:
            in_channels: Number of input channels (includes skip connection)
            out_channels: Number of output channels
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super(DecoderBlock, self).__init__()
        
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            batch_norm=batch_norm,
            dropout=dropout
        )
        
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            batch_norm=batch_norm,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention gate for focusing on relevant features in skip connections
    """
    
    def __init__(
        self,
        gate_channels: int,
        in_channels: int,
        inter_channels: int
    ):
        """
        Initialize attention gate
        
        Args:
            gate_channels: Number of channels in gate signal (from decoder)
            in_channels: Number of channels in input signal (from encoder)
            inter_channels: Number of intermediate channels
        """
        super(AttentionGate, self).__init__()
        
        # Gate signal processing
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Input signal processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Attention computation
        self.attention_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention gate
        
        Args:
            gate: Gate signal from decoder path
            skip: Skip connection from encoder path
            
        Returns:
            Attention-weighted skip connection
        """
        # Process gate signal
        gate_processed = self.gate_conv(gate)
        
        # Process input signal
        skip_processed = self.input_conv(skip)
        
        # Resize gate to match skip connection size if needed
        if gate_processed.size()[2:] != skip_processed.size()[2:]:
            gate_processed = F.interpolate(
                gate_processed,
                size=skip_processed.size()[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Compute attention weights
        attention_weights = self.attention_conv(gate_processed + skip_processed)
        
        # Apply attention weights to skip connection
        attended_skip = skip * attention_weights
        
        return attended_skip


class SpatialAttention(nn.Module):
    """
    Spatial attention module for focusing on important spatial locations
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention
        
        Args:
            kernel_size: Kernel size for convolution
        """
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(
            2, 1, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial attention
        
        Args:
            x: Input feature map
            
        Returns:
            Attention-weighted feature map
        """
        # Compute channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate statistics
        concat = torch.cat([avg_pool, max_pool], dim=1)
        
        # Compute attention weights
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel attention module for focusing on important feature channels
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Initialize channel attention
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for bottleneck
        """
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through channel attention
        
        Args:
            x: Input feature map
            
        Returns:
            Attention-weighted feature map
        """
        # Global average pooling
        avg_out = self.fc(self.avg_pool(x))
        
        # Global max pooling
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        # Apply attention
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        """
        Initialize CBAM
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CBAM
        
        Args:
            x: Input feature map
            
        Returns:
            Attention-weighted feature map
        """
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with optional attention
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_attention: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize residual block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
            use_attention: Whether to use attention mechanism
            dropout: Dropout probability
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dropout=dropout
        )
        
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            dropout=dropout
        )
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=stride, 
                bias=False
            )
        else:
            self.skip_connection = nn.Identity()
        
        # Attention mechanism
        if use_attention:
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        identity = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Apply attention
        out = self.attention(out)
        
        # Add residual connection
        out += identity
        
        return F.relu(out)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for efficient computation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1
    ):
        """
        Initialize depthwise separable convolution
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            padding: Padding size
            stride: Stride size
        """
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through depthwise separable convolution
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x 