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
    plt.imshow(image, cmap='Greys')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":

    #==== Read data ====
    X_train, labels_train, X_test, labels_test = load_dataset()
    print('x_Train',X_train)
    print('labels_train',labels_train)

    #==== One hot encode ====
    # (Use sklearn.preprocessing.OneHotEncoder)
    enc = OneHotEncoder(categorical_features=[0], handle_unknown='error', n_values='auto', sparse=True)
    enc.fit(labels_train)

    #print(enc.n_values_)
    #print(enc.feature_indices_)
    print(enc.transform(1).toarray())
    labels_train_ohe = enc.transform(labels_train).toarray()
    print(type(X_train))
    print(X_train.shape)
    print(type(labels_train))
    print(labels_train_ohe.shape)

    #==== Train ====
    # (Use sklearn.linear_model.LinearRegression)
    lmodel = linear_model.LinearRegression()
    # lmodel = linear_model.RidgeCV(alphas=[0.1, 1, 5, 10, 20, 40.5, 50, 50.5, 51])
    lmodel.fit(X_train, labels_train_ohe)

    #==== Predict ====
    # (Use sklearn.linear_model.LinearRegression)
    Y_predict_train = lmodel.predict(X_train)
    Y_predict_test = lmodel.predict(X_test)

    #==== Decode ====
    # (Use numpy.argmax)
    pred_labels_test = np.argmax(Y_predict_test, axis=1)
    pred_labels_train = np.argmax(Y_predict_train, axis=1)

    #==== Print accuracy ====
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    #==== Plot first mis-classified data ====
    # (Use the provided plot(x) function)
    for i in range(0, 10000):
        if pred_labels_test[i] != labels_test[i]:
            print(i)
            print('Truth', labels_test[i])
            print('Prediction', pred_labels_test[i])
            plot(X_test[i])
            break