import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_url, cached_download

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
                   unpad_image

from . import tario
import pandas as pd

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
            config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
            cached_download(config_file_url, cache_dir=cache_dir, force_filename=local_filename)
            print('Weights downloaded to:', os.path.join(cache_dir, local_filename))
        
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
    
def upsample_folder(directory):
    print("upsampling images from: " + directory)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    
    # Get path relative to this model.py file
    model_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(model_dir, '..', 'weights', 'RealESRGAN_x4plus.pth')
    model.load_weights(weights_path)

    up_files = []
    for root, directories, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.png'):
                file_path = os.path.join(root,filename)
                img = Image.open(file_path)
                img = np.array(img)
                sr_img = model.predict(np.array(img))

                # Replace the last directory in the path with 'UP', works for any input path
                dir_parts = os.path.normpath(file_path).split(os.sep)
                if len(dir_parts) > 1:
                    dir_parts[-2] = 'UP'
                    up_dir = os.sep.join(dir_parts[:-1])
                else:
                    up_dir = 'UP'
                os.makedirs(up_dir, exist_ok=True)
                up_path = os.path.join(up_dir, os.path.basename(file_path).replace('.png', '_up.png'))
                up_files.append(up_path)
                sr_img.save(up_path)

    return up_files

def upsample_tar(folder_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    
    # Get path relative to this model.py file
    model_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(model_dir, '..', 'weights', 'RealESRGAN_x4plus.pth')
    model.load_weights(weights_path)
    
    table = folder_path + "/" + "proj_track.csv"        
    rgnir_tar_path = folder_path+'/rgnir.tar'
    repaired_tar_path = folder_path+'/rgnir_fd.tar'
    upsampled_tar_path = folder_path+'/upsampled.tar'

    df = pd.read_csv(table)
    df['upsampled'] = False

      #if column df['cloud_island_intersection'] does not exist, create it.  This would be created by the image repair/impution step
    if 'cloud_island_intersection' not in df.columns:
        df['cloud_island_intersection'] = False
    if 'repaired' not in df.columns:
        df['repaired'] = False
    
    clean_downloads = df[(df['cloud_island_intersection'].astype(bool) == False) & (df['rgnir_download'] == True) ]
    for i, name in enumerate(clean_downloads['name'].tolist()):
        if clean_downloads['repaired'].tolist()[i] == False:
            input_tar_path = rgnir_tar_path
        else:
            input_tar_path = repaired_tar_path

        lowtar = tario.tar_io(input_tar_path)                       
        low_img = lowtar.get_from_tar(name+'_rgnir.png')
        
        low_array = np.array(low_img)[:,:,:3]
        sr_image = model.predict(np.array(low_array))
        
        hightar = tario.tar_io(upsampled_tar_path)
        hightar.save_to_tar(sr_image,name+'_sr.png',overwrite=True)
        
        df.loc[df['name'] == name, 'upsampled'] = True

        #save df to csv
        df.to_csv(table, index=False)

    return df