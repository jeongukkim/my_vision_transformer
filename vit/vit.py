"""
AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
(https://arxiv.org/pdf/2010.11929.pdf)
"""

import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 embedding_dimension=768,
                 patch_size=16,
                 num_layers=12,
                 mlp_dimension=3072,
                 num_heads=12,
                 dropout_rate=0.1,
                 num_classes=1000):
        super(VisionTransformer, self).__init__()

        self.to_patch_embedding = ImageToPatch(embedding_dimension, patch_size)

        image_height, image_width = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.add_positional_embedding = AbsolutePositionEmbeddings(num_patches + 1, embedding_dimension, dropout_rate)
        self.cls_token = nn.Parameter(torch.randn((1, 1, embedding_dimension)))

        self.transformer_encoder = nn.Sequential(*[TransformerBlocks(embedding_dimension, mlp_dimension, num_heads, dropout_rate) for _ in range(num_layers)])

        self.to_image_representation = nn.LayerNorm(embedding_dimension)
        self.cls_head = nn.Linear(embedding_dimension, num_classes)

    def forward(self, image):
        # image -> patches
        x = self.to_patch_embedding(image)

        # to patch embeddings
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.add_positional_embedding(x)

        # series of transformer blocks
        x = self.transformer_encoder(x)

        # mlp head
        image_representation = self.to_image_representation(x[:, 0, :])
        out = self.cls_head(image_representation)

        return out


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
                 embedding_dimension=768,
                 mlp_dimension=None,
                 num_heads=12,
                 dropout_rate=0.1):
        super(TransformerBlocks, self).__init__()

        hidden_dimension = mlp_dimension or embedding_dimension * 4

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
                 dropout_rate=0.1):
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


def vit_base_patch16_224(num_classes=1000):
    return VisionTransformer(num_classes=num_classes,
                             image_size=224,
                             embedding_dimension=768,
                             mlp_dimension=3072,
                             num_heads=12,
                             num_layers=12,
                             patch_size=16,
                             dropout_rate=0.1)


def vit_base_patch32_224(num_classes=1000):
    return VisionTransformer(num_classes=num_classes,
                             image_size=224,
                             embedding_dimension=768,
                             mlp_dimension=3072,
                             num_heads=12,
                             num_layers=12,
                             patch_size=32,
                             dropout_rate=0.1)


def vit_large_patch16_224(num_classes=1000):
    return VisionTransformer(num_classes=num_classes,
                             image_size=224,
                             embedding_dimension=1024,
                             mlp_dimension=4096,
                             num_heads=16,
                             num_layers=24,
                             patch_size=16,
                             dropout_rate=0.1)


def vit_large_patch32_224(num_classes=1000):
    return VisionTransformer(num_classes=num_classes,
                             image_size=224,
                             embedding_dimension=1024,
                             mlp_dimension=4096,
                             num_heads=16,
                             num_layers=24,
                             patch_size=32,
                             dropout_rate=0.1)


def vit_huge_patch14_224(num_classes=1000):
    return VisionTransformer(num_classes=num_classes,
                             image_size=224,
                             embedding_dimension=1280,
                             mlp_dimension=5120,
                             num_heads=16,
                             num_layers=32,
                             patch_size=14,
                             dropout_rate=0.1)
