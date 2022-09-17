"""
Rethinking Spatial Dimensions of Vision Transformers
(https://arxiv.org/pdf/2103.16302.pdf)
"""

import math
import torch
import torch.nn as nn


class PoolingBasedVisionTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 base_embedding_dimension=64,
                 stem_kernel_size=16,
                 stem_stride=8,
                 num_layers=[2, 6, 4],
                 num_heads=[2, 4, 8],
                 dropout_rate=0.,
                 num_classes=1000):
        super(PoolingBasedVisionTransformer, self).__init__()

        self.to_patch_embedding = ConvolutionalEmbedding(base_embedding_dimension, stem_kernel_size, stem_stride)

        image_height, image_width = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        feature_height, feature_width = (image_height - stem_kernel_size) // stem_stride + 1, (image_width - stem_kernel_size) // stem_stride + 1
        num_patches = feature_height * feature_width
        self.add_positional_embedding = AbsolutePositionEmbeddings(num_patches + 1, base_embedding_dimension, dropout_rate)
        self.cls_token = nn.Parameter(torch.randn((1, 1, base_embedding_dimension)))

        num_stage = len(num_layers)
        self.transformer_encoder = nn.ModuleList()
        for i, (num_block, num_head) in enumerate(zip(num_layers, num_heads)):
            emb_dim = base_embedding_dimension * 2 ** i
            # transformer blocks in (i+1)-th stage
            for _ in range(num_block):
                self.transformer_encoder.append(TransformerBlocks(emb_dim, num_head, dropout_rate))
            # pooling layer between stages
            if i < num_stage - 1:
                self.transformer_encoder.append(PoolingLayer(emb_dim, feature_height, feature_width))
                feature_height, feature_width = math.ceil(feature_height / 2), math.ceil(feature_width / 2)

        final_dim = base_embedding_dimension * 2 ** (num_stage-1)
        self.to_image_representation = nn.LayerNorm(final_dim)
        self.cls_head = nn.Linear(final_dim, num_classes)

    def forward(self, image):
        # image -> patches
        x = self.to_patch_embedding(image)

        # to patch embeddings
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.add_positional_embedding(x)

        # series of transformer blocks
        for encoder_block in self.transformer_encoder:
            x = encoder_block(x)

        # mlp head
        image_representation = self.to_image_representation(x[:, 0, :])
        out = self.cls_head(image_representation)

        return out


class ConvolutionalEmbedding(nn.Module):
    """
    Stem layer
    """
    def __init__(self,
                 embedding_dimension=64,
                 kernel_size=16,
                 stride=8):
        super(ConvolutionalEmbedding, self).__init__()

        self.stem = nn.Conv2d(3, embedding_dimension, kernel_size, stride)

    def forward(self, x):
        x = self.stem(x)

        B, C, *_ = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)

        return x


class AbsolutePositionEmbeddings(nn.Module):
    """
    Add learnable one-dimensional position embeddings and dropout
    """
    def __init__(self,
                 num_patches,
                 embedding_dimension=64,
                 dropout_rate=0.1):
        super(AbsolutePositionEmbeddings, self).__init__()

        self.positional_embedding = nn.Parameter(torch.randn((1, num_patches, embedding_dimension)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.positional_embedding
        x = self.dropout(x)
        return x


class TransformerBlocks(nn.Module):
    """
    Pre-Layer Normalization version of Transformer block
    (LN -> MHSA) -> residual connection -> (LN -> MLP) -> residual connection
    """
    def __init__(self,
                 embedding_dimension=64,
                 num_heads=2,
                 dropout_rate=0.1):
        super(TransformerBlocks, self).__init__()

        hidden_dimension = embedding_dimension * 4

        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.attention = MultiHeadedSelfAttention(embedding_dimension, num_heads, dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, embedding_dimension),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        identity = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = x + identity

        identity = x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = x + identity

        return x


class MultiHeadedSelfAttention(nn.Module):
    """
    Multi-Headed Self-Attention
    scaled dot-product attention
    """
    def __init__(self,
                 embedding_dimension=64,
                 num_heads=2,
                 dropout_rate=0.1):
        super(MultiHeadedSelfAttention, self).__init__()

        self.num_heads = num_heads
        head_embed_dims = embedding_dimension // num_heads
        self.scale_factor = head_embed_dims ** -0.5

        self.to_qkv = nn.Linear(embedding_dimension, embedding_dimension * 3, bias=False)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.projection_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=0)

        similarities = (q @ k.transpose(-2, -1)) * self.scale_factor
        attention_score = similarities.softmax(dim=-1)
        attention_score = self.attention_dropout(attention_score)

        x = (attention_score @ v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        x = self.projection_dropout(x)
        return x


class PoolingLayer(nn.Module):
    """
    pooling to spatial reduction
    """
    def __init__(self,
                 embedding_dimension=64,
                 feature_height=27,
                 feature_width=27):
        super(PoolingLayer, self).__init__()

        self.height, self.width = feature_height, feature_width
        self.depthwise_conv = nn.Conv2d(embedding_dimension, embedding_dimension, 3, 2, 1, groups=embedding_dimension)
        self.pointwise_conv = nn.Conv2d(embedding_dimension, embedding_dimension * 2, 1, 1, 0)

        self.linear = nn.Linear(embedding_dimension, embedding_dimension * 2)

    def forward(self, x):
        B, _, C = x.shape

        x_cls, x_img = x[:, :1, :], x[:, 1:, :]

        # spatial tokens
        x_img = x_img.permute(0, 2, 1).reshape(B, C, self.height, self.width)
        x_img = self.depthwise_conv(x_img)
        x_img = self.pointwise_conv(x_img)
        x_img = x_img.reshape(B, 2 * C, -1).permute(0, 2, 1)

        # class tokens
        x_cls = self.linear(x_cls)

        x = torch.cat([x_cls, x_img], dim=1)
        return x


def pit_ti(num_classes=1000):
    return PoolingBasedVisionTransformer(num_classes=num_classes,
                                         image_size=224,
                                         base_embedding_dimension=64,
                                         stem_kernel_size=16,
                                         stem_stride=8,
                                         num_layers=[2, 6, 4],
                                         num_heads=[2, 4, 8],)


def pit_xs(num_classes=1000):
    return PoolingBasedVisionTransformer(num_classes=num_classes,
                                         image_size=224,
                                         base_embedding_dimension=96,
                                         stem_kernel_size=16,
                                         stem_stride=8,
                                         num_layers=[2, 6, 4],
                                         num_heads=[2, 4, 8],)


def pit_s(num_classes=1000):
    return PoolingBasedVisionTransformer(num_classes=num_classes,
                                         image_size=224,
                                         base_embedding_dimension=144,
                                         stem_kernel_size=16,
                                         stem_stride=8,
                                         num_layers=[2, 6, 4],
                                         num_heads=[3, 6, 12],)


def pit_b(num_classes=1000):
    return PoolingBasedVisionTransformer(num_classes=num_classes,
                                         image_size=224,
                                         base_embedding_dimension=256,
                                         stem_kernel_size=14,
                                         stem_stride=7,
                                         num_layers=[3, 6, 4],
                                         num_heads=[4, 8, 16],)
