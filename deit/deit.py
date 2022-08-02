"""
Training data-efficient image transformers & distillation through attention
(https://arxiv.org/pdf/2012.12877.pdf)
"""

import torch
import torch.nn as nn


class DataEfficientImageTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 embedding_dimension=768,
                 patch_size=16,
                 num_layers=12,
                 num_heads=12,
                 dropout_rate=0.0,
                 num_classes=1000):
        super(DataEfficientImageTransformer, self).__init__()

        self.to_patch_embedding = ImageToPatch(embedding_dimension, patch_size)

        image_height, image_width = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.add_positional_embedding = AbsolutePositionEmbeddings(num_patches + 2, embedding_dimension, dropout_rate)
        self.cls_token = nn.Parameter(torch.randn((1, 1, embedding_dimension)))
        self.distill_token = nn.Parameter(torch.randn((1, 1, embedding_dimension)))

        self.transformer_encoder = nn.Sequential(*[TransformerBlocks(embedding_dimension, num_heads, dropout_rate) for _ in range(num_layers)])

        self.to_representation = nn.LayerNorm(embedding_dimension)
        self.cls_head = nn.Linear(embedding_dimension, num_classes)
        self.distill_head = nn.Linear(embedding_dimension, num_classes)

    def forward(self, image):
        # image -> patches
        x = self.to_patch_embedding(image)

        # to patch embeddings
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        distill_tokens = self.distill_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_tokens, x, distill_tokens], dim=1)
        x = self.add_positional_embedding(x)

        # series of transformer blocks
        x = self.transformer_encoder(x)

        # cls & distill head
        x = self.to_representation(x)
        out = self.cls_head(x[:, 0, :])
        out_distill = self.distill_head(x[:, -1, :])

        return out, out_distill


class ImageToPatch(nn.Module):
    """
    split image to non-overlapping patches then linearly project patches to embedding_dimension
    """
    def __init__(self,
                 embedding_dimension=768,
                 patch_size=16):
        super(ImageToPatch, self).__init__()

        self.patch_size = patch_size

        self.linear_projection = nn.Linear(3 * self.patch_size * self.patch_size, embedding_dimension)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * self.patch_size * self.patch_size, H // self.patch_size * W // self.patch_size)
        x = x.permute(0, 2, 1)

        return self.linear_projection(x)


class AbsolutePositionEmbeddings(nn.Module):
    """
    Add learnable one-dimensional position embeddings and dropout
    """
    def __init__(self,
                 num_patches,
                 embedding_dimension=768,
                 dropout_rate=0.0):
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
                 embedding_dimension=768,
                 num_heads=12,
                 dropout_rate=0.0):
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
                 embedding_dimension=768,
                 num_heads=12,
                 dropout_rate=0.0):
        super(MultiHeadedSelfAttention, self).__init__()

        self.num_heads = num_heads
        head_embed_dims = embedding_dimension // num_heads
        self.scale_factor = head_embed_dims ** -0.5

        self.to_qkv = nn.Linear(embedding_dimension, embedding_dimension * 3, bias=False)

        self.projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=0)

        similarities = (q @ k.transpose(-2, -1)) * self.scale_factor
        attention_score = similarities.softmax(dim=-1)

        x = (attention_score @ v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        x = self.dropout(x)
        return x


def deit_tiny_224(num_classes=1000):
    return DataEfficientImageTransformer(num_classes=num_classes,
                                         image_size=224,
                                         embedding_dimension=192,
                                         patch_size=16,
                                         num_layers=12,
                                         num_heads=3,
                                         dropout_rate=0.0)


def deit_small_224(num_classes=1000):
    return DataEfficientImageTransformer(num_classes=num_classes,
                                         image_size=224,
                                         embedding_dimension=384,
                                         patch_size=16,
                                         num_layers=12,
                                         num_heads=6,
                                         dropout_rate=0.0)


def deit_base_224(num_classes=1000):
    return DataEfficientImageTransformer(num_classes=num_classes,
                                         image_size=224,
                                         embedding_dimension=768,
                                         patch_size=16,
                                         num_layers=12,
                                         num_heads=12,
                                         dropout_rate=0.0)
