import os

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from torch.utils.data import dataset
from torchvision import transforms
import timm
import numpy as np
import random

class FeatInvDataset(dataset.Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define transformations
        self.model_input_transform = transforms.Compose([
            transforms.RandomResizedCrop(288),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.reconstruction_resize = transforms.Resize((256, 256))

        # Load feature extraction model
        self.feature_model = timm.create_model(
            'convnext_base.fb_in22k_ft_in1k',
            pretrained=True,
            features_only=False,
        )
        self.feature_model = self.feature_model.to(self.device).eval()

        # Collect image paths
        self.image_paths = self._collect_image_paths(f'./imagenet/{self.mode}/')

    @staticmethod
    def _collect_image_paths(img_dir):
        """Collect valid image paths recursively from the given directory."""
        image_paths = []
        for root, _, files in os.walk(img_dir):
            image_paths.extend(os.path.join(root, f) for f in files if f.endswith('.JPEG'))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def denormalize(img_tensor):
        """Denormalize a tensor image from ImageNet normalization."""
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        return np.clip(img, 0, 1)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        input_image = self.model_input_transform(image)
        resized_image = self.reconstruction_resize(input_image)

        # Extract features
        with torch.no_grad():
            features = self.feature_model.forward_features(input_image.unsqueeze(0).to(self.device))
            hint = features.squeeze().cpu().numpy().transpose(1, 2, 0)

        # Randomly mask the hint and adjust text
        txt = 'a high-quality, detailed, and professional image'
        if random.random() < 0.15:
            hint.fill(0)
            txt = ''

        # Prepare the representation
        jpg = (self.denormalize(resized_image).astype(np.float32) * 255 / 127.5) - 1.0

        return {"jpg": jpg, "txt": txt, "hint": hint}

dataset = FeatInvDataset(mode='train')

# Configs
resume_path = './control_featinv_convnext.ckpt'
batch_size = 8
logger_freq = 5000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# Loading model
model = create_model('/models/cldm_featinv_convnext.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(precision=32, callbacks=[logger], max_epochs=3, accelerator='gpu', gpus=1)

# Train!
trainer.fit(model, dataloader)
trainer.save_checkpoint('./control_featinv_convnext_fin.ckpt')
