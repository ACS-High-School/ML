import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sagemaker.tensorflow import TensorFlowPredictor


def test_inference(endpoint_name, num_samples):

    _, (x_test, y_test) = mnist.load_data()
    indices = random.sample(range(x_test.shape[0]), num_samples)
    images, labels = x_test[indices] / 255, y_test[indices]

    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.show()

    predictor = TensorFlowPredictor(endpoint_name)
    prediction = predictor.predict(images.reshape(num_samples, 28, 28, 1))['predictions']
    predicted_label = np.array(prediction).argmax(axis=1)

    print('The predicted labels are: {}'.format(predicted_label))