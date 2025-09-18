# predict.py
from cog import BasePredictor, Input, Path
from PIL import Image
from RealESRGAN import RealESRGAN
import torch

class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
        self,
        image: Path = Input(description="Input image to upsample"),
    ) -> Path:
        """
        Superresolve a low resolution image and get a high resolution image.
        """
        # Load image
        lr_img = Image.open(image).convert("RGB")

        # setup RealESRGAN model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)

        # Run upsampling
        hr_img = model.predict(lr_img)
        # Save the resulting mask
        output_path = "upsampled_image.png"
        hr_img.save(output_path)
        # Return mask as final output
        return Path(output_path)
