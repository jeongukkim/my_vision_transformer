"""
Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
(https://arxiv.org/pdf/2101.11986.pdf)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenToTokenVisionTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 t2t_type='performer',
                 t2t_embedding_dimension=64,
                 t2t_mlp_dimension=64,
                 t2t_patch_size=[7, 3, 3],
                 t2t_overlapping=[3, 1, 1],
                 num_layers=12,
                 embedding_dimension=384,
                 mlp_dimension=1152,
                 num_heads=14,
                 dropout_rate=0.1,
                 num_classes=1000):
        super(TokenToTokenVisionTransformer, self).__init__()

        self.to_patch_embedding = TokenToTokenModule(t2t_type, t2t_patch_size, t2t_overlapping, t2t_embedding_dimension, t2t_mlp_dimension, embedding_dimension)

        image_height, image_width = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        t2t_strides = 1
        for p, o in zip(t2t_patch_size, t2t_overlapping):
            t2t_strides *= p - o
        num_patches = (image_height // t2t_strides) * (image_width // t2t_strides)
        self.add_positional_embedding = SinusoidalPositionEmbeddings(num_patches + 1, embedding_dimension, dropout_rate)
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


class TokenToTokenModule(nn.Module):
    """
    Progressively structurize an image to tokens
    """
    def __init__(self,
                 type,
                 patch_size=[7, 3, 3],
                 overlapping=[3, 1, 1],
                 embedding_dimension=64,
                 mlp_dimension=64,
                 final_dimension=384):
        super(TokenToTokenModule, self).__init__()

        assert isinstance(type, str) and type in {'performer', 'transformer'}
        assert len(patch_size) == len(overlapping)

        num_layer = len(patch_size)

        self.first_soft_split = SplitWithOverlapping(patch_size[0], overlapping[0])
        self.blocks = nn.ModuleList()
        for i in range(1, num_layer):
            prev_dim = 3 if i == 1 else embedding_dimension
            in_dim = prev_dim * patch_size[i-1] * patch_size[i-1]
            self.blocks.append(ReStructurization(type, in_dim, embedding_dimension, mlp_dimension))
            self.blocks.append(SplitWithOverlapping(patch_size[i], overlapping[i]))
        self.projection = nn.Linear(embedding_dimension * patch_size[-1] * patch_size[-1], final_dimension)

    def forward(self, x):
        x = self.first_soft_split(x)
        for re_structurization, soft_split in zip(self.blocks[0::2], self.blocks[1::2]):
            x = re_structurization(x)
            x = soft_split(x)
        x = self.projection(x)
        return x


class SplitWithOverlapping(nn.Module):
    """
    split image to patches with overlapping
    """
    def __init__(self,
                 kernel_size,
                 overlapping):
        super(SplitWithOverlapping, self).__init__()

        self._kernel_size = kernel_size
        self._stride = kernel_size - overlapping

    def _get_padding(self, x):
        *_, H, W = x.shape

        remains_height = H % self._stride
        remains_width = W % self._stride
        padding_height = self._kernel_size - self._stride if remains_height == 0 else self._kernel_size - remains_height
        padding_width = self._kernel_size - self._stride if remains_width == 0 else self._kernel_size - remains_width

        return math.ceil(padding_height / 2), math.ceil(padding_width / 2)

    def forward(self, x):
        pad_h, pad_w = self._get_padding(x)
        return F.unfold(x, kernel_size=self._kernel_size, stride=self._stride, padding=(pad_h, pad_w)).transpose(1, 2)


class ReStructurization(nn.Module):
    """
    apply transformer blocks then reshape to image
    """
    def __init__(self,
                 type,
                 in_dimension,
                 embedding_dimension=64,
                 hidden_dimension=64):
        super(ReStructurization, self).__init__()

        if type == 'performer':
            self.attention = TokenPerformerAttention()

        if type == 'transformer':
            self.attention = TokenTransformerAttention(in_dimension, embedding_dimension)

        self.layer_norm_1 = nn.LayerNorm(in_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, embedding_dimension),
        )

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = self.attention(x)

        identity = x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = x + identity

        B, N, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(N**0.5), int(N**0.5))
        return x


# TODO: Performer 논문 읽고 구현
class TokenPerformerAttention(nn.Module):
    """
    Efficient Transformer layer at limited GPU memory
    (see RETHINKING ATTENTION WITH PERFORMERS (https://arxiv.org/pdf/2009.14794.pdf))
    """
    def __init__(self):
        super(TokenPerformerAttention, self).__init__()

        pass


class TokenTransformerAttention(nn.Module):
    """
    Multi-Headed Self-Attention
    scaled dot-product attention
    """
    def __init__(self,
                 in_dimension,
                 embedding_dimension=64,
                 num_heads=1,
                 dropout_rate=0.):
        super(TokenTransformerAttention, self).__init__()

        self.emb_dim = embedding_dimension
        self.num_heads = num_heads
        head_embed_dims = embedding_dimension // num_heads
        self.scale_factor = head_embed_dims ** -0.5

        self.to_qkv = nn.Linear(in_dimension, embedding_dimension * 3, bias=False)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.projection_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.emb_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=0)

        similarities = (q @ k.transpose(-2, -1)) * self.scale_factor
        attention_score = similarities.softmax(dim=-1)
        attention_score = self.attention_dropout(attention_score)

        x = (attention_score @ v).transpose(1, 2).reshape(B, N, self.emb_dim)
        x = self.projection(x)
        x = self.projection_dropout(x)

        x = x + v.transpose(1, 2).reshape(B, N, self.emb_dim)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Add sinusoidal position embeddings
    """
    def __init__(self,
                 num_patches,
                 embedding_dimension=384,
                 dropout_rate=0.1):
        super(SinusoidalPositionEmbeddings, self).__init__()

        get_angle = lambda t, i: t / (10000. ** (2 * (i // 2) / embedding_dimension))
        positional_embedding = torch.FloatTensor([
            [get_angle(patch_position, dim) for dim in range(embedding_dimension)]
            for patch_position in range(num_patches)])
        positional_embedding[:, 0::2] = torch.sin(positional_embedding[:, 0::2])
        positional_embedding[:, 1::2] = torch.cos(positional_embedding[:, 1::2])
        self.register_buffer('positional_embedding', positional_embedding.unsqueeze(dim=0))

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


def t2t_vit_t_14(num_classes=1000):
    return TokenToTokenVisionTransformer(num_classes=num_classes,
                                         image_size=224,
                                         t2t_type='transformer',
                                         t2t_embedding_dimension=64,
                                         t2t_mlp_dimension=64,
                                         t2t_patch_size=[7, 3, 3],
                                         t2t_overlapping=[3, 1, 1],
                                         num_layers=14,
                                         embedding_dimension=384,
                                         mlp_dimension=1152,
                                         num_heads=6,
                                         dropout_rate=0.)
