#!/usr/bin/env python

import torch.nn as nn

from detectron2.config import CfgNode
from detectron2.utils.logger import setup_logger

from .transformer import TextTransformer as _TextTransformer


logger = setup_logger(name='prompter')


class PrompterTextTransformer(_TextTransformer):

    def __init__(self, *args, **kwargs):
        # classnames = kwargs.pop('classnames', None)
        n_ctx = kwargs.pop('additional_context_lenght', 8)
        # assert classnames, 'Class Names are required'
        assert n_ctx > 0, 'Context lenght must be greater then 0'

        super(PrompterTextTransformer, self).__init__(*args, **kwargs)

        transformer_width = kwargs.get('width')

        ctx_vectors = torch.empty(n_ctx, transformer_width, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # prompt_prefix = ' '.join(['X'] * n_ctx)

        # classnames = [name.replace('_', ' ') for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]

        # logger.info(f'Initial context: "{prompt_prefix}"')
        # logger.info(f'Number of context words (tokens): {n_ctx}')
        
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        embedding = self.token_embedding(self.tokenized_prompts).type(dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + n_ctx, :]
                
    def forward(self, prompts: List[torch.Tensor]) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.text_projection

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        texts = self.update_text(texts)
        text_tokens = self.tokenize(texts)
        embedding = self.token_embedding(text_tokens)

        prefix, suffix = embedding[:, :1, :], embedding[:, 1 + self.ctx.shape[0], :]
        prompts = torch.cat([prefix, self.ctx, suffix], dim=1)
        return self.forward(prompts)

    def update_text(self, texts: List[str], placeholder: str = 'X') -> List[str]:
        assert isinstance(placeholder, str) and len(placeholder) > 0
        prefix = ' '.join(placeholder) * n_ctx
        return [prefix + ' ' + text.replace('_', ' ') + '.' for text in texts]


class Prompter(nn.Module):

    def __init__(self, ):
        pass

    def forward(self):
        pass


def build_prompter(cfg: CfgNode):
    """
    Builds the promp learning module
    """
    pass

