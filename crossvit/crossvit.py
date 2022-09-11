"""
CrossViT: Cross-Attention Multi-scale Vision Transformer for Image Classification
(https://arxiv.org/pdf/2103.14899.pdf)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionMultiScaleVisionTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 small_embedding_dimension=96,
                 large_embedding_dimension=192,
                 small_patch_size=12,
                 large_patch_size=16,
                 num_encoder_layers=3,
                 num_small_branch_encoder_layers=1,
                 num_large_branch_encoder_layers=4,
                 num_cross_attention_layers=1,
                 mlp_dimension_ratio=4,
                 num_heads=3,
                 dropout_rate=0.1,
                 num_classes=1000):
        super(CrossAttentionMultiScaleVisionTransformer, self).__init__()

        image_height, image_width = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)

        self.to_small_patch_embedding = ImageToPatch(small_embedding_dimension, small_patch_size)
        num_small_patches = math.ceil(image_height / small_patch_size) * math.ceil(image_width / small_patch_size)
        self.add_positional_embedding_to_small = AbsolutePositionEmbeddings(num_small_patches + 1, small_embedding_dimension, dropout_rate)
        self.small_cls_token = nn.Parameter(torch.randn((1, 1, small_embedding_dimension)))

        self.to_large_patch_embedding = ImageToPatch(large_embedding_dimension, large_patch_size)
        num_large_patches = (image_height // large_patch_size) * (image_width // large_patch_size)
        self.add_positional_embedding_to_large = AbsolutePositionEmbeddings(num_large_patches + 1, large_embedding_dimension, dropout_rate)
        self.large_cls_token = nn.Parameter(torch.randn((1, 1, large_embedding_dimension)))

        self.multi_scale_transformer_encoder = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.multi_scale_transformer_encoder.append(
                MultiScaleTransformerBlocks(small_embedding_dimension,
                                            large_embedding_dimension,
                                            num_small_branch_encoder_layers,
                                            num_large_branch_encoder_layers,
                                            num_cross_attention_layers,
                                            num_heads,
                                            mlp_dimension_ratio,
                                            dropout_rate)
                )

        self.to_image_representation_small = nn.LayerNorm(small_embedding_dimension)
        self.to_image_representation_large = nn.LayerNorm(large_embedding_dimension)
        self.cls_head_small = nn.Linear(small_embedding_dimension, num_classes)
        self.cls_head_large = nn.Linear(large_embedding_dimension, num_classes)

    def forward(self, image):
        # image -> patch embeddings
        ## small patch
        x_s = self.to_small_patch_embedding(image)
        cls_tokens_s = self.small_cls_token.repeat(image.shape[0], 1, 1)
        x_s = torch.cat([cls_tokens_s, x_s], dim=1)
        x_s = self.add_positional_embedding_to_small(x_s)

        ## large patch
        x_l = self.to_large_patch_embedding(image)
        cls_tokens_l = self.large_cls_token.repeat(image.shape[0], 1, 1)
        x_l = torch.cat([cls_tokens_l, x_l], dim=1)
        x_l = self.add_positional_embedding_to_large(x_l)

        # series of multi-scale transformer blocks
        for transformer_block in self.multi_scale_transformer_encoder:
            x_s, x_l = transformer_block(x_s, x_l)

        # mlp head
        ## small patch
        image_representation_s = self.to_image_representation_small(x_s[:, 0, :])
        out_s = self.cls_head_small(image_representation_s)

        ## large patch
        image_representation_l = self.to_image_representation_large(x_l[:, 0, :])
        out_l = self.cls_head_large(image_representation_l)

        return out_s + out_l


class ImageToPatch(nn.Module):
    """
    split image to non-overlapping patches then linearly project patches to embedding_dimension
    """
    def __init__(self,
                 embedding_dimension=192,
                 patch_size=16):
        super(ImageToPatch, self).__init__()

        self.patch_size = patch_size

        self.linear_projection = nn.Linear(3 * self.patch_size * self.patch_size, embedding_dimension)

    def forward(self, x):
        B, C, H, W = x.shape

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            H = math.ceil(H / self.patch_size) * self.patch_size
            W = math.ceil(W / self.patch_size) * self.patch_size
            x = F.interpolate(x, (H, W))

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
                 embedding_dimension=192,
                 dropout_rate=0.1):
        super(AbsolutePositionEmbeddings, self).__init__()

        self.positional_embedding = nn.Parameter(torch.randn((1, num_patches, embedding_dimension)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.positional_embedding
        x = self.dropout(x)
        return x


class MultiScaleTransformerBlocks(nn.Module):
    """
    Transformer blocks for learning multi-scale features with cross-attention
    """
    def __init__(self,
                 small_embedding_dimension=96,
                 large_embedding_dimension=192,
                 num_small_branch_encoder_layers=1,
                 num_large_branch_encoder_layers=4,
                 num_cross_attention_layers=1,
                 num_heads=3,
                 mlp_dimension_ratio=4,
                 dropout_rate=0.1):
        super(MultiScaleTransformerBlocks, self).__init__()

        self.small_branch = nn.Sequential(*[TransformerBlocks(small_embedding_dimension, mlp_dimension_ratio, num_heads, dropout_rate) for _ in range(num_small_branch_encoder_layers)])
        self.large_branch = nn.Sequential(*[TransformerBlocks(large_embedding_dimension, mlp_dimension_ratio, num_heads, dropout_rate) for _ in range(num_large_branch_encoder_layers)])

        self.multi_scale_fusion = nn.ModuleList()
        for _ in range(num_cross_attention_layers):
            self.multi_scale_fusion.append(CrossAttentionModule(small_embedding_dimension, large_embedding_dimension, num_heads, dropout_rate))
            self.multi_scale_fusion.append(CrossAttentionModule(large_embedding_dimension, small_embedding_dimension, num_heads, dropout_rate))

    def forward(self, x_s, x_l):
        x_s = self.small_branch(x_s)
        x_l = self.large_branch(x_l)

        x_cls_s, x_img_s = x_s[:, :1, :], x_s[:, 1:, :]
        x_cls_l, x_img_l = x_l[:, :1, :], x_l[:, 1:, :]

        for cross_attention_small_branch, cross_attention_large_branch in zip(self.multi_scale_fusion[0::2], self.multi_scale_fusion[1::2]):
            x_cls_s = cross_attention_small_branch(x_cls_s, x_img_l)
            x_cls_l = cross_attention_large_branch(x_cls_l, x_img_s)

        x_s = torch.cat([x_cls_s, x_img_s], dim=1)
        x_l = torch.cat([x_cls_l, x_img_l], dim=1)

        return x_s, x_l


class TransformerBlocks(nn.Module):
    """
    Pre-Layer Normalization version of Transformer block
    (LN -> MHSA) -> residual connection -> (LN -> MLP) -> residual connection
    """
    def __init__(self,
                 embedding_dimension=192,
                 mlp_dimension_ratio=4,
                 num_heads=3,
                 dropout_rate=0.1):
        super(TransformerBlocks, self).__init__()

        hidden_dimension = embedding_dimension * mlp_dimension_ratio

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
                 embedding_dimension=192,
                 num_heads=3,
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


class CrossAttentionModule(nn.Module):
    """
    Cross-Attention among CLS token from one branch and patch tokens from another branch
    """
    def __init__(self,
                 class_token_embedding_dimension,
                 patch_embedding_dimension,
                 num_heads=3,
                 dropout_rate=0.1):
        super(CrossAttentionModule, self).__init__()

        self.projection = nn.Linear(class_token_embedding_dimension, patch_embedding_dimension)

        self.num_heads = num_heads
        head_embed_dims = patch_embedding_dimension // num_heads
        self.scale_factor = head_embed_dims ** -0.5

        self.to_q = nn.Linear(patch_embedding_dimension, patch_embedding_dimension, bias=False)
        self.to_kv = nn.Linear(patch_embedding_dimension, patch_embedding_dimension * 2, bias=False)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.back_projection = nn.Linear(patch_embedding_dimension, class_token_embedding_dimension)
        self.back_projection_dropout = nn.Dropout(dropout_rate)

    def forward(self, x_cls, x_img):
        x_cls = self.projection(x_cls)
        identity = x_cls
        x = torch.cat([x_cls, x_img], dim=1)

        B, N, C = x.shape
        q = self.to_q(x_cls).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.to_kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.chunk(2, dim=0)

        similarities = (q @ k.transpose(-2, -1)) * self.scale_factor
        attention_score = similarities.softmax(dim=-1)
        attention_score = self.attention_dropout(attention_score)

        x_cls = (attention_score @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = identity + x_cls

        x_cls = self.back_projection(x_cls)
        x_cls = self.back_projection_dropout(x_cls)

        return x_cls


def crossvit_ti(num_classes=1000):
    return CrossAttentionMultiScaleVisionTransformer(num_classes=num_classes,
                                                     image_size=224,
                                                     small_embedding_dimension=96,
                                                     large_embedding_dimension=192,
                                                     small_patch_size=12,
                                                     large_patch_size=16,
                                                     num_encoder_layers=3,
                                                     num_small_branch_encoder_layers=1,
                                                     num_large_branch_encoder_layers=4,
                                                     num_cross_attention_layers=1,
                                                     mlp_dimension_ratio=4,
                                                     num_heads=3,
                                                     dropout_rate=0.)


def crossvit_s(num_classes=1000):
    return CrossAttentionMultiScaleVisionTransformer(num_classes=num_classes,
                                                     image_size=224,
                                                     small_embedding_dimension=192,
                                                     large_embedding_dimension=384,
                                                     small_patch_size=12,
                                                     large_patch_size=16,
                                                     num_encoder_layers=3,
                                                     num_small_branch_encoder_layers=1,
                                                     num_large_branch_encoder_layers=4,
                                                     num_cross_attention_layers=1,
                                                     mlp_dimension_ratio=4,
                                                     num_heads=6,
                                                     dropout_rate=0.)


def crossvit_b(num_classes=1000):
    return CrossAttentionMultiScaleVisionTransformer(num_classes=num_classes,
                                                     image_size=224,
                                                     small_embedding_dimension=384,
                                                     large_embedding_dimension=768,
                                                     small_patch_size=12,
                                                     large_patch_size=16,
                                                     num_encoder_layers=3,
                                                     num_small_branch_encoder_layers=1,
                                                     num_large_branch_encoder_layers=4,
                                                     num_cross_attention_layers=1,
                                                     mlp_dimension_ratio=4,
                                                     num_heads=12,
                                                     dropout_rate=0.)


def crossvit_9(num_classes=1000):
    return CrossAttentionMultiScaleVisionTransformer(num_classes=num_classes,
                                                     image_size=224,
                                                     small_embedding_dimension=128,
                                                     large_embedding_dimension=256,
                                                     small_patch_size=12,
                                                     large_patch_size=16,
                                                     num_encoder_layers=3,
                                                     num_small_branch_encoder_layers=1,
                                                     num_large_branch_encoder_layers=3,
                                                     num_cross_attention_layers=1,
                                                     mlp_dimension_ratio=3,
                                                     num_heads=4,
                                                     dropout_rate=0.)


def crossvit_15(num_classes=1000):
    return CrossAttentionMultiScaleVisionTransformer(num_classes=num_classes,
                                                     image_size=224,
                                                     small_embedding_dimension=192,
                                                     large_embedding_dimension=384,
                                                     small_patch_size=12,
                                                     large_patch_size=16,
                                                     num_encoder_layers=3,
                                                     num_small_branch_encoder_layers=1,
                                                     num_large_branch_encoder_layers=5,
                                                     num_cross_attention_layers=1,
                                                     mlp_dimension_ratio=3,
                                                     num_heads=6,
                                                     dropout_rate=0.)


def crossvit_18(num_classes=1000):
    return CrossAttentionMultiScaleVisionTransformer(num_classes=num_classes,
                                                     image_size=224,
                                                     small_embedding_dimension=224,
                                                     large_embedding_dimension=448,
                                                     small_patch_size=12,
                                                     large_patch_size=16,
                                                     num_encoder_layers=3,
                                                     num_small_branch_encoder_layers=1,
                                                     num_large_branch_encoder_layers=6,
                                                     num_cross_attention_layers=1,
                                                     mlp_dimension_ratio=3,
                                                     num_heads=7,
                                                     dropout_rate=0.)
