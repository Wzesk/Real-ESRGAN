import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_url, hf_hub_download

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
                   unpad_image


HF_MODELS = {
    2: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}


class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=scale
        )
        
    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path) and download:
            assert self.scale in [2,4,8], 'You can download models only with scales: 2, 4, 8'
            config = HF_MODELS[self.scale]
            cache_dir = os.path.dirname(model_path)
            local_filename = os.path.basename(model_path)
            downloaded_path = hf_hub_download(repo_id=config['repo_id'], filename=config['filename'], cache_dir=cache_dir, local_files_only=False)
            # Move the downloaded file to the expected location if needed
            expected_path = os.path.join(cache_dir, local_filename)
            if downloaded_path != expected_path:
                import shutil
                shutil.copy2(downloaded_path, expected_path)
            print('Weights downloaded to:', expected_path)
        
        loadnet = torch.load(model_path)
        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        self.model.eval()
        self.model.to(self.device)
        
    @torch.cuda.amp.autocast()
    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()

        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i+batch_size])), 0)

        sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(
            np_sr_image, padded_image_shape=padded_size_scaled, 
            target_shape=scaled_image_shape, padding_size=padding * scale
        )
        sr_img = (np_sr_image*255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size*scale)
        sr_img = Image.fromarray(sr_img)

        return sr_img


def upsample_folder(input_folder, output_folder=None, scale=4, device=None):
    """
    Upsample all images in a folder using RealESRGAN.
    
    Args:
        input_folder (str): Path to input folder containing images
        output_folder (str, optional): Path to output folder. If None, creates 'upsampled' subfolder
        scale (int): Upscaling factor (2, 4, or 8)
        device (str, optional): Device to use ('cuda' or 'cpu'). Auto-detects if None
        
    Returns:
        list: List of output image paths
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Set up output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'upsampled')
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize model
    model = RealESRGAN(device, scale=scale)
    weights_path = f'weights/RealESRGAN_x{scale}.pth'
    model.load_weights(weights_path, download=True)
    
    # Process images
    upsampled_images = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            
            # Create output filename
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_x{scale}{ext}"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                # Load and process image
                image = Image.open(input_path).convert('RGB')
                sr_image = model.predict(image)
                sr_image.save(output_path)
                upsampled_images.append(output_path)
                print(f"Upsampled: {filename} -> {output_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                
    return upsampled_images
