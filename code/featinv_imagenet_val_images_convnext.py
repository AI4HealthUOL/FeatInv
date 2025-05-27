import os

from PIL import Image

import torch
from torchvision.transforms import transforms

from featinv_reconstructor_convnext import InputReconstructor
import timm

def reconstruct():
    # Initialize feature model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f_model = timm.create_model(
        'convnext_base.fb_in22k_ft_in1k',
        pretrained=True,
        features_only=False,
    ).eval().to(device)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(288),
    ])
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    additional_transform = transforms.Resize((256, 256))

    val_dir = './imagenet/val/'
    save_root = './reconstructed_10perClass_convnext'

    for class_folder in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        save_dir = os.path.join(save_root, class_folder)
        os.makedirs(save_dir, exist_ok=True)

        # Process first 10 images in the class folder
        for image_file in os.listdir(class_path)[:10]:
            image_path = os.path.join(class_path, image_file)
            try:
                # Load and process image
                original_image = Image.open(image_path).convert('RGB')
                cropped_image = transform(original_image)
                cropped_image_save_path = os.path.join(save_dir, image_file)

                # Reconstruct and save results
                input_tensor = transform_to_tensor(cropped_image).unsqueeze(0).to(device)
                reconstructor = InputReconstructor(f_model)
                reconstructor.reconstruct_input_original(
                    input_tensor, os.path.join(save_dir, f'reconstructed_{image_file}')
                )
                additional_transform(cropped_image).save(cropped_image_save_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
reconstruct()
