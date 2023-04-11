#!/usr/bin/env python

from typing import List

import numpy as np
import torch
import torch.nn as nn

from detectron2.config import CfgNode

import clip
from clip.model import Transformer, LayerNorm


class TextTransformer(Transformer):
    def __init__(self, vocab_size: int, context_length: int, embed_dim: int, **kwargs):
        super(TextTransformer, self).__init__(**kwargs)
        self.context_length = context_length
        transformer_width = kwargs.get('width')

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ln_final = LayerNorm(transformer_width)
        self.initialize_parameters()

        self._embed_dim = embed_dim

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.width**-0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width**-0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.width**-0.5)

    def forward(self, text: List[torch.Tensor]):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text(self, text: List[str]) -> torch.Tensor:
        return self.forward(self.tokenized(text_tokens))

    def tokenize(self, text: List[str]) -> torch.Tensor:
        # TODO: Fix the device
        return clip.tokenize(text).cuda()
        
    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def dtype(self) -> str:
        return self.resblocks[0].attn.out_proj.weight.dtype


def build_text_encoder(cfg: CfgNode) -> TextTransformer:

    from .clip_model import build_attention_mask

    vocab_size = cfg.MODEL.CLIP.TEXT_ENCODER.VOCAB_SIZE
    context_length = cfg.MODEL.CLIP.TEXT_ENCODER.CONTEXT_LENGTH
    embed_dim = cfg.MODEL.CLIP.TEXT_ENCODER.EMBED_DIM
    transformer_width = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_WIDTH
    transformer_layers = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_LAYERS
    transformer_heads = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_HEADS
    transformer = TextTransformer(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_dim=embed_dim,
        width=transformer_width,
        layers=transformer_layers,
        heads=transformer_heads,
        attn_mask=build_attention_mask(cfg),
    )

    return transformer
