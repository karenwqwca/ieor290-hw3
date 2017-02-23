from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder


def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0

    X_train = np.asarray(X_train).reshape(60000, 784)
    labels_train = np.asarray(labels_train).reshape(60000, 1)
    X_test = np.asarray(X_test).reshape(10000, 784)
    labels_test = np.asarray(labels_test).reshape(10000, 1)

    return X_train, labels_train, X_test, labels_test

def plot(x):
    image = x.reshape(28, 28)
    plt.imshow(image, cmap='grey')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":

    #==== Read data ====
    X_train, labels_train, X_test, labels_test = load_dataset()

    #==== One hot encode ====
    # (Use sklearn.preprocessing.OneHotEncoder)

    #==== Train ====
    # (Use sklearn.linear_model.LinearRegression)

    #==== Predict ====
    # (Use sklearn.linear_model.LinearRegression)

    #==== Decode ====
    # (Use numpy.argmax)

    #==== Print accuracy ====
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    #==== Plot first mis-classified data ====
    # (Use the provided plot(x) function)