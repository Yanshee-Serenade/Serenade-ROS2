import datetime

import cv2
from PIL import Image

from ..config import config


def cv2_to_pil(cv_image):
    """Converts a cv2 image to a PIL image and saves it.

    Args:
        cv_image (numpy.ndarray): The cv2 image to convert.

    Returns:
        tuple: A tuple containing the PIL image and the path where it was saved.
    """
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image_rgb)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_path = config.get_image_path(timestamp)
    pil_image.save(image_save_path)
    print(
        f"[{timestamp}] 💾 Saved processed image to: {image_save_path} "
        f"(size: {pil_image.size}, mode: {pil_image.mode})"
    )
    return (pil_image, image_save_path)
