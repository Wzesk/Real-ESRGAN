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
        lr_array = np.array(lr_img)

        # Run upsampling
        hr_array = re.predict(lr_array)
        hr_img = Image.fromarray(hr_array)

        # Save the resulting mask
        output_path = "upsampled_image.png"
        hr_img.save(output_path)

        # Return mask as final output
        return Path(output_path)
