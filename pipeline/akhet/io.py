"""Image input: path, PIL image or numpy array -> PIL RGB (EXIF orientation applied for files)."""
import numpy as np
from PIL import Image, ImageOps


def load_image(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = np.stack([x] * 3, -1)
        return Image.fromarray(x[..., :3].astype(np.uint8), "RGB")
    im = Image.open(x)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")
