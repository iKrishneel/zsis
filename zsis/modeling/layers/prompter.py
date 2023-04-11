#!/usr/bin/env python

from typing import List, Type

import torch
import torch.nn as nn

from detectron2.config import CfgNode
from detectron2.utils.logger import setup_logger

from .transformer import TextTransformer as _TextTransformer


logger = setup_logger(name='prompter')


T: Type[torch.Tensor] = torch.Tensor


class PrompterTextTransformer(_TextTransformer):

    def __init__(self, *args, **kwargs):
        n_ctx = kwargs.pop('additional_context_lenght', 8)
        assert n_ctx > 0, 'Context lenght must be greater then 0'

        super(PrompterTextTransformer, self).__init__(*args, **kwargs)

        transformer_width = kwargs.get('width')

        ctx_vectors = torch.empty(n_ctx, transformer_width, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
                
    def forward(self, prompts: T, text_tokens: T) -> T:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection

    def encode_text(self, texts: List[str]) -> T:
        texts = self.update_text(texts)
        text_tokens = self.tokenize(texts)
        embedding = self.token_embedding(text_tokens)

        prefix, suffix = embedding[:, :1, :], embedding[:, 1 + self.ctx.shape[0]:, :]

        ctx = self.ctx
        ctx = ctx[None].expand(len(texts), -1, -1) if ctx.dim() == 2 else ctx
        
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return self.forward(prompts, text_tokens)

    def update_text(self, texts: List[str], placeholder: str = 'X') -> List[str]:
        assert isinstance(placeholder, str) and len(placeholder) > 0
        prefix = ' '.join(placeholder * self.ctx.shape[0])
        return [prefix + ' ' + text.replace('_', ' ') + '.' for text in texts]


def build_prompter_transformer(cfg: CfgNode) -> PrompterTextTransformer:
    """
    Builds the promp learning module
    """
    from .clip_model import build_attention_mask
    
    vocab_size = cfg.MODEL.CLIP.TEXT_ENCODER.VOCAB_SIZE
    context_length = cfg.MODEL.CLIP.TEXT_ENCODER.CONTEXT_LENGTH
    embed_dim = cfg.MODEL.CLIP.TEXT_ENCODER.EMBED_DIM
    transformer_width = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_WIDTH
    transformer_layers = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_LAYERS
    transformer_heads = cfg.MODEL.CLIP.TEXT_ENCODER.TRANSFORMER_HEADS
    transformer = PrompterTextTransformer(
        vocab_size=vocab_size,
        context_length=context_length,
        embed_dim=embed_dim,
        width=transformer_width,
        layers=transformer_layers,
        heads=transformer_heads,
        attn_mask=build_attention_mask(cfg),
    )

    return transformer
