import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train = pd.read_csv("./temp/sign_mnist_train/sign_mnist_train_clean.csv")
test = pd.read_csv("./temp/sign_mnist_test/sign_mnist_test.csv")

# Representations of the data
plt.figure(figsize=(10, 10))
sns.set_style("darkgrid")
sns.countplot(x="label", data=train, palette="viridis")

# Distribution of the labels
""" plt.savefig('./temp/label_count.png')
image = Image.open('./temp/label_count.png')
image.show() """

y_train = train["label"]
y_test = test["label"]
del train["label"]  # Equivalent to drop
del test["label"]

# Check for null and missing values
# train.info()
# test.dtypes()
# labels = np.array(labels)
# np.unique(labels)
# train.isnull().values.any()
# train[train.duplicated()]

# Drop Nan & duplicated
train = train.drop([317, 487, 595, 689, 727, 802, 861], axis=0)

# Normalize the data
train = train.astype(str).astype(int)
train = train / 255.0
test = test.astype(str).astype(int)
test = test / 255.0

print(train.head())
