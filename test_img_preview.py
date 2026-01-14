import requests
import datetime
from PIL import Image
from io import BytesIO

docker_ip = "localhost"
port = 51121

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
processed_path = f"image_{timestamp}.jpg"

image_response = requests.get(f"http://{docker_ip}:{port}/image")
pil_image = Image.open(BytesIO(image_response.content))
pil_image.save(processed_path)