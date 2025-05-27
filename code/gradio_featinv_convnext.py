import config
from torch import nn

import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    zero_module
)

model = create_model('/models/cldm_featinv_convnext.yaml')
hint_channels = 1024
intermediate_channels = 512
model.control_model.input_hint_block.block = TimestepEmbedSequential(
    nn.Upsample(size=(32, 32), mode='bilinear'),
    conv_nd(model.control_model.dims, hint_channels, intermediate_channels, 3, padding=1),
    nn.SiLU(),
    conv_nd(model.control_model.dims, intermediate_channels, intermediate_channels, 3, padding=1),
    nn.SiLU(),
    zero_module(
        conv_nd(model.control_model.dims, intermediate_channels, model.control_model.model_channels, 3, padding=1)),
)
model.load_state_dict(load_state_dict('./control_featinv_convnext_fin.ckpt', location='cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
ddim_sampler = DDIMSampler(model)


def process(feature_map_np, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed,
            eta):
    with torch.no_grad():
        feature_map = feature_map_np
        control = torch.from_numpy(feature_map.copy()).float()
        control = control.to('cuda')
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, 32, 32)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                    [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results
