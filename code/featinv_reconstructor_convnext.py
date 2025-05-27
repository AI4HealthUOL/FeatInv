import os
import numpy as np
import torch
from PIL import Image

import gradio_featinv_convnext


class InputReconstructor:
    def __init__(self, feature_model):
        self.feature_model = feature_model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_model.to(self.device)

    def reconstruct_input_original(self, transformed_image, save_path):
        # Extract features
        with torch.no_grad():
            features = self.feature_model.forward_features(transformed_image.to(self.device)).squeeze().cpu().numpy()

        # Prepare input for reconstruction
        input_data = np.expand_dims(features, axis=0)

        # Process features with reconstruction utility
        gallery = gradio_featinv_convnext.process(
            input_data,
            prompt='a high-quality, detailed, and professional image',
            a_prompt='',
            n_prompt='',
            num_samples=1,
            ddim_steps=50,
            guess_mode=False,
            strength=1.0,
            scale=1.75,
            seed=100,
            eta=0.0
        )

        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        # Save images from the gallery
        for i, image_array in enumerate(gallery):
            image = Image.fromarray(image_array)
            image.save(f"{save_path}_{i}.png")

        return gallery
