# Real-ESRGAN for Littoral Pipeline

PyTorch implementation of a Real-ESRGAN model trained on custom dataset. This model shows better results on faces compared to the original version. It is also easier to integrate this model into your projects.

> This is not an official implementation. We partially use code from the [original repository](https://github.com/xinntao/Real-ESRGAN)

Real-ESRGAN is an upgraded [ESRGAN](https://arxiv.org/abs/1809.00219) trained with pure synthetic data is capable of enhancing details while removing annoying artifacts for common real-world images.

## Littoral Pipeline Integration

This module serves as **Step 6** in the Littoral shoreline analysis pipeline, providing super-resolution enhancement for cloud-free satellite imagery before segmentation. It implements a standardized interface that allows for easy integration and future model updates.

### Pipeline Context
```
Cloud-free Images → [Real-ESRGAN Upsampling] → High Resolution Images → Segmentation → ...
```

### Interface
- **Input**: Cloud-free satellite imagery at native resolution (typically 10m/pixel for Sentinel-2)
- **Output**: Super-resolved images with enhanced spatial detail (4x resolution increase)
- **Function**: `RealESRGAN.upsample_folder()`
- **Technology**: Real-ESRGAN generative adversarial networks for image super-resolution

### Usage in Littoral Pipeline
```python
import sys
sys.path.append('/path/to/Real-ESRGAN')
from RealESRGAN.model import RealESRGAN
import RealESRGAN.model as re

# Upsample folder of cloud-free images
input_folder = "/path/to/clear/images"
upsampled_images = re.upsample_folder(input_folder)

# Output images saved with "_up" suffix in UP folder
print(f"Upsampled {len(upsampled_images)} images")
```

## Original Real-ESRGAN Information

You can try it in [google colab](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing)

- [Paper (Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data)](https://arxiv.org/abs/2107.10833)
- [Original implementation](https://github.com/xinntao/Real-ESRGAN)
- [Huggingface 🤗](https://huggingface.co/sberbank-ai/Real-ESRGAN)

### Installation

```bash
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```

### Usage

---

Basic usage:

```python
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
```

## Credits and Attribution

### Real-ESRGAN Citation
When using this super-resolution module in research, please cite the original Real-ESRGAN paper:

```bibtex
@misc{wang2021realesrgan,
    title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    year={2021},
    eprint={2107.10833},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

### Implementation Credits
- **Original Implementation**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Sberbank AI Implementation**: [sberbank-ai/Real-ESRGAN](https://github.com/sberbank-ai/Real-ESRGAN)
- **ESRGAN Foundation**: Wang, Xintao, et al. "ESRGAN: Enhanced super-resolution generative adversarial networks." ECCV 2018.

## Contributors

This module integration for the Littoral project has had numerous contributors, including:

**Core Development**: Walter Zesk, Tishya Chhabra, Leandra Tejedor, Philip Ndikum

**Project Leadership**: Sarah Dole, Skylar Tibbits, Peter Stempel

## Reference

This project draws extensive inspiration from the [CoastSat Project](https://github.com/kvos/CoastSat):

Vos K., Splinter K.D., Harley M.D., Simmons J.A., Turner I.L. (2019). CoastSat: a Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery. Environmental Modelling and Software. 122, 104528. https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)