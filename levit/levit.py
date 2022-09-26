"""
LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference
(https://arxiv.org/pdf/2104.01136.pdf)
"""

import math
import torch
import torch.nn as nn


class LeVisionTransformer(nn.Module):
    """
    Transformer with pooling, similar to the LeNet architecture
    """
    def __init__(self,
                 image_size=224,
                 embedding_dimension=[128, 256, 384],
                 num_layers=[2, 3, 4],
                 num_heads=[4, 6, 8],
                 qk_embedding_dimension=16,
                 dropout_rate=0.,
                 num_classes=1000):
        super(LeVisionTransformer, self).__init__()

        num_stem_layers = 4
        self.to_patch_embedding = PatchEmbedding(num_stem_layers, embedding_dimension[0], 3)

        image_height, image_width = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        feature_size = list(map(lambda x: x // 2 ** num_stem_layers, [image_height, image_width]))
        self.transformer_encoder = nn.ModuleList()
        for i, num_block in enumerate(num_layers):
            for _ in range(num_block):
                self.transformer_encoder.append(TransformerBlocks(feature_size, embedding_dimension[i], qk_embedding_dimension, num_heads[i], dropout_rate))
            if i < len(num_layers) - 1:
                self.transformer_encoder.append(PoolingBlocks(feature_size, embedding_dimension[i], embedding_dimension[i+1], qk_embedding_dimension, dropout_rate))
                feature_size = list(map(lambda x: math.ceil(x / 2), feature_size))

        self.cls_head = nn.Linear(embedding_dimension[-1], num_classes)

    def forward(self, image):
        # image -> patches
        x = self.to_patch_embedding(image)

        # series of transformer blocks
        for encoder_block in self.transformer_encoder:
            x = encoder_block(x)

        # mlp head
        image_representation = x.mean(dim=(2, 3))
        out = self.cls_head(image_representation)

        return out


class PatchEmbedding(nn.Module):
    """
    Stem layer with 4 conv2d
    """
    def __init__(self,
                 num_layers,
                 encoder_input_dimension=128,
                 kernel_size=3):
        super(PatchEmbedding, self).__init__()

        in_dim = 3
        stem_network = []
        for i in range(num_layers)[::-1]:
            out_dim = encoder_input_dimension // 2 ** i
            stem_network.append(ConvNorm(in_dim, out_dim, kernel_size, 2, 1))
            in_dim = out_dim

        self.stem = nn.Sequential(*stem_network)

    def forward(self, x):
        x = self.stem(x)
        return x


class ConvNorm(nn.Module):
    """
    Conv -> BatchNorm
    In 'Normalization layers and activations' of Section 4.2.,
    for Levit, each convolution is followed by a batch normalization.
    """
    def __init__(self,
                 in_dimension,
                 out_dimension,
                 kernel_size=3,
                 stride=2,
                 padding=1,):
        super(ConvNorm, self).__init__()

        self.net = nn.Sequential(*[
            nn.Conv2d(in_dimension, out_dimension, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_dimension)
        ])

    def forward(self, x):
        x = self.net(x)
        return x


class TransformerBlocks(nn.Module):
    """
    Transformer block with convolution
    (MHSA) -> residual connection -> (MLP) -> residual connection
    - MLP expansion ratio = 2
    - use Hardswish instead of GELU following 'Normalization layers and activations' of Section 4.2.
    - use 1x1 Conv2d instead of Linear
    """
    def __init__(self,
                 feature_size,
                 embedding_dimension=128,
                 qk_embedding_dimension=16,
                 num_heads=4,
                 dropout_rate=0.):
        super(TransformerBlocks, self).__init__()

        reduce_act_map = False
        hidden_dimension = embedding_dimension * 2

        self.attention = MultiHeadedSelfAttention(feature_size, embedding_dimension, embedding_dimension, qk_embedding_dimension, num_heads, reduce_act_map, dropout_rate)
        self.feed_forward = nn.Sequential(
            ConvNorm(embedding_dimension, hidden_dimension, 1, 1, 0),
            nn.Hardswish(),
            nn.Dropout(dropout_rate),
            ConvNorm(hidden_dimension, embedding_dimension, 1, 1, 0),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        identity = x
        x = self.attention(x)
        x = x + identity

        identity = x
        x = self.feed_forward(x)
        x = x + identity

        return x


class MultiHeadedSelfAttention(nn.Module):
    """
    Multi-Headed Self-Attention
    scaled dot-product attention
    - use 1x1 conv instead of Linear
    """
    def __init__(self,
                 feature_size,
                 in_dimension=128,
                 out_dimension=128,
                 qk_embedding_dimension=16,
                 num_heads=4,
                 q_subsampling=False,
                 dropout_rate=0.):
        super(MultiHeadedSelfAttention, self).__init__()

        if q_subsampling: # if pooling attention,
            out_feature_scale = 0.5
            key_value_dim_ratio = 4
            conv_stride = 2
        else:
            out_feature_scale = 1
            key_value_dim_ratio = 2
            conv_stride = 1

        self.num_heads = num_heads
        self.out_feature_size = list(map(lambda x: math.ceil(out_feature_scale * x), feature_size))
        self.qk_head_embed_dims = qk_embedding_dimension
        self.v_head_embed_dims = key_value_dim_ratio * qk_embedding_dimension
        self.scale_factor = qk_embedding_dimension ** -0.5

        self.to_q = ConvNorm(in_dimension, self.qk_head_embed_dims * num_heads, 1, conv_stride, 0)
        self.to_k = ConvNorm(in_dimension, self.qk_head_embed_dims * num_heads, 1, 1, 0)
        self.to_v = ConvNorm(in_dimension, self.v_head_embed_dims * num_heads, 1, 1, 0)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.attention_bias_table = nn.Parameter(torch.zeros((self.num_heads, feature_size[0] * feature_size[1])))
        q_coords_h = torch.arange(0, feature_size[0], step=conv_stride)
        q_coords_w = torch.arange(0, feature_size[1], step=conv_stride)
        k_coords_h = torch.arange(0, feature_size[0])
        k_coords_w = torch.arange(0, feature_size[1])
        q_coords = torch.stack(torch.meshgrid([q_coords_h, q_coords_w])).flatten(start_dim=1)
        k_coords = torch.stack(torch.meshgrid([k_coords_h, k_coords_w])).flatten(start_dim=1)
        relative_coords = (q_coords[:, :, None] - k_coords[:, None, :]).abs()
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] *= feature_size[1]
        relative_position_index = relative_coords.sum(dim=-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.non_linear = nn.Hardswish()
        self.projection = ConvNorm(self.v_head_embed_dims * num_heads, out_dimension, 1, 1, 0)
        self.projection_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x: 4-d tensor [B, C, H, W]
        """
        B, _, H, W = x.shape
        q = self.to_q(x).reshape(B, self.num_heads, self.qk_head_embed_dims, -1).permute(0, 1, 3, 2)
        k = self.to_k(x).reshape(B, self.num_heads, self.qk_head_embed_dims, -1)
        v = self.to_v(x).reshape(B, self.num_heads, self.v_head_embed_dims, -1).permute(0, 1, 3, 2)

        similarities = (q @ k) * self.scale_factor
        attention_bias = self.attention_bias_table[:, self.relative_position_index.view(-1)].reshape(1, self.num_heads, self.out_feature_size[0] * self.out_feature_size[1], H * W)
        similarities = similarities + attention_bias
        attention_score = similarities.softmax(dim=-1)
        attention_score = self.attention_dropout(attention_score)

        x = (attention_score @ v).transpose(-2, -1).reshape(B, self.num_heads * self.v_head_embed_dims, *self.out_feature_size)
        x = self.non_linear(x)
        x = self.projection(x)
        x = self.projection_dropout(x)
        return x


class PoolingBlocks(nn.Module):
    """
    Transformer block with convolution for pooling
    """
    def __init__(self,
                 feature_size,
                 in_dimension=128,
                 out_dimension=256,
                 qk_embedding_dimension=16,
                 dropout_rate=0.):
        super(PoolingBlocks, self).__init__()

        reduce_act_map = True
        num_heads = in_dimension // qk_embedding_dimension
        hidden_dimension = out_dimension * 2

        self.attention = MultiHeadedSelfAttention(feature_size, in_dimension, out_dimension, qk_embedding_dimension, num_heads, reduce_act_map, dropout_rate)
        self.feed_forward = nn.Sequential(
            ConvNorm(out_dimension, hidden_dimension, 1, 1, 0),
            nn.Hardswish(),
            nn.Dropout(dropout_rate),
            ConvNorm(hidden_dimension, out_dimension, 1, 1, 0),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.attention(x)

        identity = x
        x = self.feed_forward(x)
        x = x + identity

        return x


def levit_128_s(num_classes=1000):
    return LeVisionTransformer(num_classes=num_classes,
                               image_size=224,
                               embedding_dimension=[128, 256, 384],
                               num_layers=[2, 3, 4],
                               num_heads=[4, 6, 8],
                               qk_embedding_dimension=16,
                               dropout_rate=0.)


def levit_128(num_classes=1000):
    return LeVisionTransformer(num_classes=num_classes,
                               image_size=224,
                               embedding_dimension=[128, 256, 384],
                               num_layers=[4, 4, 4],
                               num_heads=[4, 8, 12],
                               qk_embedding_dimension=16,
                               dropout_rate=0.)


def levit_192(num_classes=1000):
    return LeVisionTransformer(num_classes=num_classes,
                               image_size=224,
                               embedding_dimension=[192, 288, 384],
                               num_layers=[4, 4, 4],
                               num_heads=[3, 5, 6],
                               qk_embedding_dimension=32,
                               dropout_rate=0.)


def levit_256(num_classes=1000):
    return LeVisionTransformer(num_classes=num_classes,
                               image_size=224,
                               embedding_dimension=[256, 384, 512],
                               num_layers=[4, 4, 4],
                               num_heads=[4, 6, 8],
                               qk_embedding_dimension=32,
                               dropout_rate=0.)


def levit_384(num_classes=1000):
    return LeVisionTransformer(num_classes=num_classes,
                               image_size=224,
                               embedding_dimension=[384, 512, 768],
                               num_layers=[4, 4, 4],
                               num_heads=[6, 9, 12],
                               qk_embedding_dimension=32,
                               dropout_rate=0.1)


if __name__ == "__main__":
    import torchsummary
    
    model = levit_128_s().cuda()
    torchsummary.summary(model, (3, 224, 224))