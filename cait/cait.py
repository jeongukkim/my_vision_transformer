"""
Going deeper with Image Transformers
(https://arxiv.org/pdf/2103.17239.pdf)
"""

import torch
import torch.nn as nn


class ClassAttentionImageTransformers(nn.Module):
    def __init__(self,
                 image_size=224,
                 embedding_dimension=192,
                 patch_size=16,
                 layer_scale_init_weight=1e-5,
                 num_encoder_layers=24,
                 num_decoder_layers=2,
                 num_heads=4,
                 dropout_rate=0.0,
                 num_classes=1000):
        super(ClassAttentionImageTransformers, self).__init__()

        self.to_patch_embedding = ImageToPatch(embedding_dimension, patch_size)

        image_height, image_width = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.add_positional_embedding = AbsolutePositionEmbeddings(num_patches, embedding_dimension, dropout_rate)
        self.cls_token = nn.Parameter(torch.randn((1, 1, embedding_dimension)))

        self.transformer_encoder = nn.Sequential(*[EncoderBlocks(layer_scale_init_weight, embedding_dimension, num_heads, dropout_rate) for _ in range(num_encoder_layers)])
        self.transformer_decoder = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.transformer_decoder.append(DecoderBlocks(layer_scale_init_weight, embedding_dimension, num_heads, dropout_rate))

        self.to_image_representation = nn.LayerNorm(embedding_dimension)
        self.cls_head = nn.Linear(embedding_dimension, num_classes)

    def forward(self, image):
        # image -> patches
        x = self.to_patch_embedding(image)
        x = self.add_positional_embedding(x)

        # self-attention stage
        x = self.transformer_encoder(x)

        # class-attention stage
        x_cls = self.cls_token.repeat(x.shape[0], 1, 1)
        for decoder_block in self.transformer_decoder:
            x_cls = decoder_block(x_cls, x)

        # cls head
        image_representation = self.to_image_representation(x_cls)
        out = self.cls_head(image_representation)

        return out


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
                 dropout_rate=0.0):
        super(AbsolutePositionEmbeddings, self).__init__()

        self.positional_embedding = nn.Parameter(torch.randn((1, num_patches, embedding_dimension)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.positional_embedding
        x = self.dropout(x)
        return x


class EncoderBlocks(nn.Module):
    """
    Pre-Layer Normalization version of Transformer Encoder block
    (LN -> MHSA) -> residual connection -> (LN -> MLP) -> residual connection
    """
    def __init__(self,
                 layer_scale_init_weight,
                 embedding_dimension=192,
                 num_heads=4,
                 dropout_rate=0.0):
        super(EncoderBlocks, self).__init__()

        hidden_dimension = embedding_dimension * 4

        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.attention = MultiHeadedSelfAttention(embedding_dimension, num_heads, dropout_rate)
        self.layer_scale_1 = LayerScale(layer_scale_init_weight, embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, embedding_dimension),
            nn.Dropout(dropout_rate)
        )
        self.layer_scale_2 = LayerScale(layer_scale_init_weight, embedding_dimension)

    def forward(self, x):
        identity = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.layer_scale_1(x)
        x = x + identity

        identity = x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = self.layer_scale_2(x)
        x = x + identity

        return x


class MultiHeadedSelfAttention(nn.Module):
    """
    Multi-Headed Self-Attention
    scaled dot-product attention
    """
    def __init__(self,
                 embedding_dimension=192,
                 num_heads=4,
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


class LayerScale(nn.Module):
    """
    Per-channel weighting on the output of residual blocks
    """
    def __init__(self,
                 epsilon,
                 embedding_dimension=192):
        super(LayerScale, self).__init__()

        initial_weight = torch.zeros((embedding_dimension)).fill_(epsilon)
        self.per_channel_multiplication = nn.Parameter(initial_weight)

    def forward(self, x):
        return x * self.per_channel_multiplication


class DecoderBlocks(nn.Module):
    """
    Pre-Layer Normalization version of Transformer Decoder block
    Only the class embedding is updated
    (LN -> MHCA) -> residual connection -> (LN -> MLP) -> residual connection
    """
    def __init__(self,
                 layer_scale_init_weight,
                 embedding_dimension=192,
                 num_heads=4,
                 dropout_rate=0.0):
        super(DecoderBlocks, self).__init__()

        hidden_dimension = embedding_dimension * 4

        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.attention = MultiHeadedClassAttention(embedding_dimension, num_heads, dropout_rate)
        self.layer_scale_1 = LayerScale(layer_scale_init_weight, embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, embedding_dimension),
            nn.Dropout(dropout_rate)
        )
        self.layer_scale_2 = LayerScale(layer_scale_init_weight, embedding_dimension)

    def forward(self, class_embeddings, patch_embeddings):
        identity = class_embeddings
        x = torch.cat([class_embeddings, patch_embeddings], dim=1)
        x = self.layer_norm_1(x)
        x_cls = self.attention(x)
        x_cls = self.layer_scale_1(x_cls)
        x_cls = x_cls + identity

        identity = x_cls
        x_cls = self.layer_norm_2(x_cls)
        x_cls = self.feed_forward(x_cls)
        x_cls = self.layer_scale_2(x_cls)
        x_cls = x_cls + identity

        return x_cls


class MultiHeadedClassAttention(nn.Module):
    """
    Multi-Headed Class-Attention
    scaled dot-product attention
    """
    def __init__(self,
                 embedding_dimension=192,
                 num_heads=4,
                 dropout_rate=0.0):
        super(MultiHeadedClassAttention, self).__init__()

        self.num_heads = num_heads
        head_embed_dims = embedding_dimension // num_heads
        self.scale_factor = head_embed_dims ** -0.5

        self.to_q = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.to_kv = nn.Linear(embedding_dimension, embedding_dimension * 2, bias=False)

        self.projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        q = self.to_q(x[:, 0, :]).unsqueeze(dim=1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.to_kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.chunk(2, dim=0)

        similarities = (q @ k.transpose(-2, -1)) * self.scale_factor
        attention_score = similarities.softmax(dim=-1)

        x_cls = (attention_score @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.projection(x_cls)
        x_cls = self.dropout(x_cls)
        return x_cls


def cait_xxs(num_classes=1000, image_size=224, layer_scale_init_weight=1e-5, num_encoder_layers=24):
    return ClassAttentionImageTransformers(num_classes=num_classes,
                                           image_size=image_size,
                                           embedding_dimension=192,
                                           layer_scale_init_weight=layer_scale_init_weight,
                                           patch_size=16,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=2,
                                           num_heads=4,
                                           dropout_rate=0.0)

def cait_xxs24_224(num_classes=1000):
    return cait_xxs(num_classes=num_classes, image_size=224, layer_scale_init_weight=1e-5, num_encoder_layers=24)


def cait_xxs36_224(num_classes=1000):
    return cait_xxs(num_classes=num_classes, image_size=224, layer_scale_init_weight=1e-6, num_encoder_layers=36)


def cait_xxs24_384(num_classes=1000):
    return cait_xxs(num_classes=num_classes, image_size=384, layer_scale_init_weight=1e-5, num_encoder_layers=24)


def cait_xxs36_384(num_classes=1000):
    return cait_xxs(num_classes=num_classes, image_size=384, layer_scale_init_weight=1e-6, num_encoder_layers=36)


def cait_xs(num_classes=1000, image_size=224, layer_scale_init_weight=1e-5, num_encoder_layers=24):
    return ClassAttentionImageTransformers(num_classes=num_classes,
                                           image_size=image_size,
                                           embedding_dimension=288,
                                           layer_scale_init_weight=layer_scale_init_weight,
                                           patch_size=16,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=2,
                                           num_heads=6,
                                           dropout_rate=0.0)

def cait_xs24_224(num_classes=1000):
    return cait_xs(num_classes=num_classes, image_size=224, layer_scale_init_weight=1e-5, num_encoder_layers=24)


def cait_xs36_224(num_classes=1000):
    return cait_xs(num_classes=num_classes, image_size=224, layer_scale_init_weight=1e-6, num_encoder_layers=36)


def cait_xs24_384(num_classes=1000):
    return cait_xs(num_classes=num_classes, image_size=384, layer_scale_init_weight=1e-5, num_encoder_layers=24)


def cait_xs36_384(num_classes=1000):
    return cait_xs(num_classes=num_classes, image_size=384, layer_scale_init_weight=1e-6, num_encoder_layers=36)


def cait_s(num_classes=1000, image_size=224, layer_scale_init_weight=1e-5, num_encoder_layers=24):
    return ClassAttentionImageTransformers(num_classes=num_classes,
                                           image_size=image_size,
                                           embedding_dimension=384,
                                           layer_scale_init_weight=layer_scale_init_weight,
                                           patch_size=16,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=2,
                                           num_heads=8,
                                           dropout_rate=0.0)


def cait_s24_224(num_classes=1000):
    return cait_s(num_classes=num_classes, image_size=224, layer_scale_init_weight=1e-5, num_encoder_layers=24)


def cait_s36_224(num_classes=1000):
    return cait_s(num_classes=num_classes, image_size=224, layer_scale_init_weight=1e-6, num_encoder_layers=36)


def cait_s48_224(num_classes=1000):
    return cait_s(num_classes=num_classes, image_size=224, layer_scale_init_weight=1e-6, num_encoder_layers=48)


def cait_s24_384(num_classes=1000):
    return cait_s(num_classes=num_classes, image_size=384, layer_scale_init_weight=1e-5, num_encoder_layers=24)


def cait_s36_384(num_classes=1000):
    return cait_s(num_classes=num_classes, image_size=384, layer_scale_init_weight=1e-6, num_encoder_layers=36)


def cait_s48_384(num_classes=1000):
    return cait_s(num_classes=num_classes, image_size=384, layer_scale_init_weight=1e-6, num_encoder_layers=48)


def cait_m(num_classes=1000, image_size=224, layer_scale_init_weight=1e-5, num_encoder_layers=24):
    return ClassAttentionImageTransformers(num_classes=num_classes,
                                           image_size=image_size,
                                           embedding_dimension=768,
                                           layer_scale_init_weight=layer_scale_init_weight,
                                           patch_size=16,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=2,
                                           num_heads=16,
                                           dropout_rate=0.0)


def cait_m24_224(num_classes=1000):
    return cait_m(num_classes=num_classes, image_size=224, layer_scale_init_weight=1e-5, num_encoder_layers=24)


def cait_m36_224(num_classes=1000):
    return cait_m(num_classes=num_classes, image_size=224, layer_scale_init_weight=1e-6, num_encoder_layers=36)


def cait_m24_384(num_classes=1000):
    return cait_m(num_classes=num_classes, image_size=384, layer_scale_init_weight=1e-5, num_encoder_layers=24)


def cait_m36_384(num_classes=1000):
    return cait_m(num_classes=num_classes, image_size=384, layer_scale_init_weight=1e-6, num_encoder_layers=36)
