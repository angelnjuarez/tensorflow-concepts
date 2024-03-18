import json
import codecs
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

url = '../temp/sign_mnist_json/data.json'

data_json = []
with codecs.open(url, 'r', 'utf-8') as js:
    for line in js:
        data_json.append(json.loads(line))

print("{} imagenes cargadas".format(len(data_json)))

images = []
for data in data_json:
    response = requests.get(data['content'])
    img = np.array(Image.open(BytesIO(response.content)))
    images.append([img, data["label"]])

plt.imshow(images[0][0].reshape(28, 28))
plt.savefig('../temp/imagenJS.png')
image = Image.open('../temp/imagenJS.png')
image.show()
