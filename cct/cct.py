"""
Escaping the Big Data Paradigm with Compact Transformer
(https://arxiv.org/pdf/2104.05704.pdf)
"""

import torch
import torch.nn as nn


class CompactConvolutionalTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 embedding_dimension=128,
                 num_conv_layers=2,
                 conv_kernel_size=3,
                 conv_stride=1,
                 num_layers=2,
                 mlp_dimension=128,
                 num_heads=2,
                 dropout_rate=0.1,
                 num_classes=1000):
        super(CompactConvolutionalTransformer, self).__init__()

        self.to_patch_embedding = ConvolutionalTokenizer(embedding_dimension, num_conv_layers, conv_kernel_size, conv_stride)

        image_height, image_width = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        num_patches = (image_height // (2 * conv_stride) ** num_conv_layers) * (image_width // (2 * conv_stride) ** num_conv_layers)
        self.add_positional_embedding = AbsolutePositionEmbeddings(num_patches, embedding_dimension, dropout_rate)

        self.transformer_encoder = nn.Sequential(*[TransformerBlocks(embedding_dimension, mlp_dimension, num_heads, dropout_rate) for _ in range(num_layers)])

        self.to_image_representation = SequencePooling(embedding_dimension)
        self.cls_head = nn.Linear(embedding_dimension, num_classes)

    def forward(self, image):
        # image -> patches
        x = self.to_patch_embedding(image)

        # to patch embeddings
        x = self.add_positional_embedding(x)

        # series of transformer blocks
        x = self.transformer_encoder(x)

        # mlp head
        image_representation = self.to_image_representation(x)
        out = self.cls_head(image_representation)

        return out


class ConvolutionalTokenizer(nn.Module):
    """
    Use Convolutional Tokenizer in order to introduce inductive bias
    """
    def __init__(self,
                 embedding_dimension=128,
                 num_layers=2,
                 kernel_size=3,
                 stride=1):
        super(ConvolutionalTokenizer, self).__init__()

        tokenizer_blocks = []
        for i in range(num_layers):
            prev_dim = 3 if i == 0 else embedding_dimension
            tokenizer_blocks.append(nn.Conv2d(prev_dim, embedding_dimension, kernel_size, stride, kernel_size // 2, bias=False))
            tokenizer_blocks.append(nn.ReLU())
            tokenizer_blocks.append(nn.MaxPool2d(2, 2))
        self.tokenizer = nn.Sequential(*tokenizer_blocks)

    def forward(self, x):
        x = self.tokenizer(x)

        B, C, *_ = x.shape
        x = x.view(B, C, -1)
        return x.transpose(1, 2)


class AbsolutePositionEmbeddings(nn.Module):
    """
    Add learnable one-dimensional position embeddings and dropout
    """
    def __init__(self,
                 num_patches,
                 embedding_dimension=128,
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
                 embedding_dimension=128,
                 mlp_dimension=128,
                 num_heads=2,
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
                 embedding_dimension=128,
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


class SequencePooling(nn.Module):
    """
    Attention-based method which pools over the output sequence of tokens
    """
    def __init__(self,
                 embedding_dimension=128):
        super(SequencePooling, self).__init__()

        self.linear = nn.Linear(embedding_dimension, 1)

    def forward(self, x):
        B = x.shape[0]

        attention_score = self.linear(x).softmax(dim=1)
        x = (attention_score.transpose(1, 2) @ x).reshape(B, -1)
        return x


def cct_2_3x2_32(num_classes=1000):
    return CompactConvolutionalTransformer(num_classes=num_classes,
                                           image_size=32,
                                           embedding_dimension=128,
                                           num_conv_layers=2,
                                           conv_kernel_size=3,
                                           conv_stride=1,
                                           num_layers=2,
                                           mlp_dimension=128,
                                           num_heads=2,
                                           dropout_rate=0.1)


def cct_7_3x2_32(num_classes=1000):
    return CompactConvolutionalTransformer(num_classes=num_classes,
                                           image_size=32,
                                           embedding_dimension=256,
                                           num_conv_layers=2,
                                           conv_kernel_size=3,
                                           conv_stride=1,
                                           num_layers=7,
                                           mlp_dimension=512,
                                           num_heads=4,
                                           dropout_rate=0.1)


def cct_14_7x2_224(num_classes=1000):
    return CompactConvolutionalTransformer(num_classes=num_classes,
                                           image_size=224,
                                           embedding_dimension=384,
                                           num_conv_layers=2,
                                           conv_kernel_size=7,
                                           conv_stride=2,
                                           num_layers=14,
                                           mlp_dimension=1152,
                                           num_heads=6,
                                           dropout_rate=0.1)
