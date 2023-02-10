#!/usr/bin/env python

import os
import click
import clip
import numpy as np
import torch
from collections import OrderedDict
from tabulate import tabulate


@click.command()
@click.option('--clip_model', type=str, required=True)
@click.option('--save_path', type=str, default=os.getcwd())
@click.option('--fpn/--no-fpn', type=bool, default=False)
def main(clip_model: str, save_path: str, fpn: bool):
    assert clip_model in clip.available_models(), f'{clip_model} not found, available are {clip.available_models()}'

    filename = os.path.join(os.environ['HOME'], f'.cache/clip/{clip_model}.pt')
    if os.path.isfile(filename):
        print(f'Loading CLIP weights from {filename}')
        state_dict = torch.jit.load(filename, map_location='cpu').state_dict()
    else:
        print(f'Loading {clip_model}')
        model, _ = clip.load(clip_model)
        state_dict = model.state_dict()

        input_resolution = model.visual.input_resolution
        context_length = model.context_length
        vocab_size = model.vocab_size

        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)

    print('converting weights...')
    param_names = ['token_embedding', 'positional_embedding', 'text_projection', 'logit_scale', 'ln_final']
    new_state_dict = OrderedDict()

    keys = []
    for key in state_dict.keys():
        if 'transformer' in key:
            new_key = key
        elif 'visual' in key:
            if 'attn' in key:
                new_key = key.replace('visual.', '')
            else:
                name = 'backbone.bottom_up.model.model' if fpn else 'backbone.model.model'
                new_key = key.replace('visual', name)
        elif any([name in key for name in param_names]):
            new_key = 'transformer.' + key
            # new_key = key
        else:
            print(f'\033[031mUnknown key {key} \033[0m')
            continue

        keys.append([key, new_key])
        new_state_dict[new_key] = state_dict[key].float()

    print(tabulate(keys, tablefmt='grid'))
    if not '.pth' in save_path:
        save_path = os.path.join(save_path, f'{clip_model}.pth')

    torch.save(new_state_dict, save_path)
    print(f'Saving converted weights to {save_path}')


if __name__ == '__main__':
    main()
