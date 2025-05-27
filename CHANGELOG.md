# Changelog

## Changes to the Original ControlNet Project for FeatInv Integration

This project based on [ControlNet](https://github.com/lllyasviel/ControlNet) includes modifications and additions to support FeatInv. All changes below reflect deviations from the upstream repository, licensed under Apache 2.0.

### Modified Files

- `cldm/cldm.py`: Adjusted the hint input to support FeatInv requirements.
- `cldm/logger.py`: Updated logging behavior to align with FeatInv workflows.

### New Files

- `featinv_imagenet_val_images_convnext.py`: Script for generating reconstructed images from ImageNet validation samples.
- `featinv_reconstructor_convnext.py`: Handles input reconstruction tasks as part of the FeatInv system.
- `gradio_featinv_convnext.py`: Gradio interface for testing FeatInv using a ConvNeXt backbone.
- `tool_add_control_featinv.py`: Script to create and prepare models necessary for FeatInv.
- `tutorial_train_featinv.py`: Training script to train the FeatInv model based on the ControlNet architecture.
