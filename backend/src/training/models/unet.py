"""
Attention-Enhanced U-Net Architecture for Nail Segmentation

Implements a U-Net model with attention mechanisms, optimized for small datasets
and nail segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

import sys
from pathlib import Path

# Add training root to path for imports
training_root = Path(__file__).parent.parent
sys.path.insert(0, str(training_root))

from models.components import EncoderBlock, DecoderBlock, AttentionGate
from utils.config import TrainingConfig


class AttentionUNet(nn.Module):
    """
    Attention-Enhanced U-Net for nail segmentation
    
    Features:
    - Encoder-decoder architecture with skip connections
    - Attention gates for feature refinement
    - Configurable depth and channels
    - Dropout and batch normalization
    - Optimized for small datasets
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize Attention U-Net
        
        Args:
            config: Training configuration containing model parameters
        """
        super(AttentionUNet, self).__init__()
        
        self.config = config
        self.input_channels = config.input_channels
        self.output_channels = config.output_channels
        self.base_channels = config.base_channels
        self.depth = config.depth
        self.use_attention = config.attention
        self.dropout = config.dropout
        self.batch_norm = config.batch_norm
        
        # Calculate channel sizes for each level
        self.channels = [self.base_channels * (2 ** i) for i in range(self.depth + 1)]
        
        # Create encoder path
        self.encoder_blocks = self._create_encoder()
        
        # Create decoder path
        self.decoder_blocks = self._create_decoder()
        
        # Create attention gates if enabled
        if self.use_attention:
            self.attention_gates = self._create_attention_gates()
        
        # Final output layer
        self.final_conv = nn.Conv2d(
            self.base_channels, 
            self.output_channels, 
            kernel_size=1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output segmentation mask of shape (B, output_channels, H, W)
        """
        # Store encoder features for skip connections
        encoder_features = []
        
        # Encoder path
        current = x
        for i, encoder_block in enumerate(self.encoder_blocks[:-1]):
            current = encoder_block(current)
            encoder_features.append(current)
            current = F.max_pool2d(current, kernel_size=2, stride=2)
        
        # Bottleneck
        current = self.encoder_blocks[-1](current)
        
        # Decoder path
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Get corresponding encoder feature first to get target size
            encoder_idx = len(encoder_features) - 1 - i
            skip_connection = encoder_features[encoder_idx]
            
            # Upsample to match skip connection size
            current = F.interpolate(
                current, 
                size=skip_connection.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # Apply attention gate if enabled
            if self.use_attention and i < len(self.attention_gates):
                skip_connection = self.attention_gates[i](
                    gate=current, 
                    skip=skip_connection
                )
            
            # Concatenate skip connection
            current = torch.cat([current, skip_connection], dim=1)
            
            # Apply decoder block
            current = decoder_block(current)
        
        # Final output
        output = self.final_conv(current)
        
        # Apply sigmoid for binary segmentation
        output = torch.sigmoid(output)
        
        return output
    
    def _create_encoder(self) -> nn.ModuleList:
        """Create encoder blocks"""
        encoder_blocks = nn.ModuleList()
        
        # First block (input -> base_channels)
        encoder_blocks.append(
            EncoderBlock(
                in_channels=self.input_channels,
                out_channels=self.channels[0],
                dropout=self.dropout,
                batch_norm=self.batch_norm
            )
        )
        
        # Subsequent blocks
        for i in range(1, len(self.channels)):
            encoder_blocks.append(
                EncoderBlock(
                    in_channels=self.channels[i-1],
                    out_channels=self.channels[i],
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                )
            )
        
        return encoder_blocks
    
    def _create_decoder(self) -> nn.ModuleList:
        """Create decoder blocks"""
        decoder_blocks = nn.ModuleList()
        
        # Decoder blocks (reverse order)
        for i in range(len(self.channels) - 2, -1, -1):
            # Input channels = current level + skip connection
            in_channels = self.channels[i+1] + self.channels[i]
            out_channels = self.channels[i]
            
            decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                )
            )
        
        return decoder_blocks
    
    def _create_attention_gates(self) -> nn.ModuleList:
        """Create attention gates for skip connections"""
        attention_gates = nn.ModuleList()
        
        for i in range(len(self.channels) - 2, -1, -1):
            gate_channels = self.channels[i+1]  # From decoder
            in_channels = self.channels[i]      # From encoder
            inter_channels = in_channels // 2
            
            attention_gates.append(
                AttentionGate(
                    gate_channels=gate_channels,
                    in_channels=in_channels,
                    inter_channels=inter_channels
                )
            )
        
        return attention_gates
    
    def _initialize_weights(self) -> None:
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model architecture information
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_info = {
            'model_name': 'AttentionUNet',
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'base_channels': self.base_channels,
            'depth': self.depth,
            'attention_enabled': self.use_attention,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'channels_per_level': self.channels
        }
        
        return model_info
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for visualization
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps at different levels
        """
        feature_maps = {}
        encoder_features = []
        
        # Encoder path
        current = x
        feature_maps['input'] = current
        
        for i, encoder_block in enumerate(self.encoder_blocks[:-1]):
            current = encoder_block(current)
            feature_maps[f'encoder_{i}'] = current
            encoder_features.append(current)
            current = F.max_pool2d(current, kernel_size=2, stride=2)
        
        # Bottleneck
        current = self.encoder_blocks[-1](current)
        feature_maps['bottleneck'] = current
        
        # Decoder path
        for i, decoder_block in enumerate(self.decoder_blocks):
            encoder_idx = len(encoder_features) - 1 - i
            skip_connection = encoder_features[encoder_idx]
            
            current = F.interpolate(
                current, 
                size=skip_connection.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            if self.use_attention and i < len(self.attention_gates):
                skip_connection = self.attention_gates[i](
                    gate=current, 
                    skip=skip_connection
                )
            
            current = torch.cat([current, skip_connection], dim=1)
            current = decoder_block(current)
            feature_maps[f'decoder_{i}'] = current
        
        # Final output
        output = self.final_conv(current)
        output = torch.sigmoid(output)
        feature_maps['output'] = output
        
        return feature_maps
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters for transfer learning"""
        for param in self.encoder_blocks.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters"""
        for param in self.encoder_blocks.parameters():
            param.requires_grad = True
    
    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract encoder features for feature analysis
        
        Args:
            x: Input tensor
            
        Returns:
            List of encoder feature tensors
        """
        features = []
        current = x
        
        for i, encoder_block in enumerate(self.encoder_blocks[:-1]):
            current = encoder_block(current)
            features.append(current)
            current = F.max_pool2d(current, kernel_size=2, stride=2)
        
        # Bottleneck
        current = self.encoder_blocks[-1](current)
        features.append(current)
        
        return features


class LightweightUNet(AttentionUNet):
    """
    Lightweight version of U-Net for faster training and inference
    """
    
    def __init__(self, config: TrainingConfig):
        # Modify config for lightweight version
        lightweight_config = TrainingConfig()
        lightweight_config.__dict__.update(config.__dict__)
        lightweight_config.base_channels = config.base_channels // 2
        lightweight_config.depth = min(config.depth, 3)
        
        super().__init__(lightweight_config)


class DeepUNet(AttentionUNet):
    """
    Deeper version of U-Net for better feature extraction
    """
    
    def __init__(self, config: TrainingConfig):
        # Modify config for deep version
        deep_config = TrainingConfig()
        deep_config.__dict__.update(config.__dict__)
        deep_config.depth = config.depth + 1
        deep_config.dropout = min(config.dropout + 0.1, 0.5)
        
        super().__init__(deep_config)


def create_model(config: TrainingConfig, model_variant: str = "standard") -> AttentionUNet:
    """
    Factory function to create U-Net models
    
    Args:
        config: Training configuration
        model_variant: Model variant ('standard', 'lightweight', 'deep')
        
    Returns:
        U-Net model instance
    """
    if model_variant == "lightweight":
        return LightweightUNet(config)
    elif model_variant == "deep":
        return DeepUNet(config)
    else:
        return AttentionUNet(config)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params 