# predict.py
from cog import BasePredictor, Input, Path
from PIL import Image
import RealESRGAN.model as re

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
        # Run upsampling
        hr_img = re.predict(lr_img)
        # Save the resulting mask
        output_path = "upsampled_image.png"
        hr_img.save(output_path)
        # Return mask as final output
        return Path(output_path)
