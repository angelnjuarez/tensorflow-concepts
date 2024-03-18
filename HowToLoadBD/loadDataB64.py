import json
import base64
from PIL import Image

url = '../temp/sign_mnist_base64/data.json'

with open(url) as f:
    data = json.load(f)

base64_img = data['b'].encode('utf-8')
path = '../temp/imagen.png'
with open(path, "wb") as f_save:
    decoded = base64.decodebytes(base64_img)
    f_save.write(decoded)

image = Image.open(path)
image.show()
