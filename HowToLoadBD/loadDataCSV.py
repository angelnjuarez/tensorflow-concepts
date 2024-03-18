import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

train = pd.read_csv('../temp/sign_mnist_train/sign_mnist_train.csv')
test = pd.read_csv('../temp/sign_mnist_test/sign_mnist_test.csv')

print(train.head())
print(train.shape)

labels = train['label'].values
train.drop('label', axis=1, inplace=True)

print(train.head())

images = train.values

plt.imshow(images[0].reshape(28, 28))
plt.savefig('../temp/imagenCSV.png')
image = Image.open('../temp/imagenCSV.png')
image.show()
