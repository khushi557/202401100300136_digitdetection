# 202401100300136_digitdetection
# Image Classification and Digit Detection using Machine Learning and Deep Learning

## Project: 202401100300136_Digit_detection

## Overview
This project focuses on *Image Classification and Digit Detection* using *Machine Learning and Deep Learning* techniques. It utilizes the *MNIST dataset* for handwritten digit recognition and applies deep learning methodologies to achieve high accuracy.

## Dataset
The project is based on the *MNIST dataset, which contains **60,000 training images* and *10,000 test images* of handwritten digits (0-9), each of size *28x28 pixels*.

## Project Structure

|-- 202401100300136_Digit_detection/
    |-- model/
        |-- trained_model.h5  # Saved trained model
    |-- data/
        |-- mnist.npz  # MNIST dataset (loaded from Keras)
    |-- notebooks/
        |-- digit_detection.ipynb  # Google Colab notebook
    |-- results/
        |-- submission.csv  # Predicted labels for test data
    |-- README.md  # Project documentation
    |-- requirements.txt  # Dependencies


## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    bash
    git clone https://github.com/your-username/202401100300136_Digit_detection.git
    cd 202401100300136_Digit_detection
    

2. Install dependencies:
    bash
    pip install -r requirements.txt
    

3. Run the Jupyter Notebook (or open it in Google Colab):
    bash
    jupyter notebook
    

## Model Architecture
- *Input Layer*: 784 neurons (28x28 pixels flattened)
- *Hidden Layer 1: 512 neurons, **ReLU activation, **Dropout (0.2)*
- *Hidden Layer 2: 512 neurons, **ReLU activation, **Dropout (0.2)*
- *Output Layer: 10 neurons (for digits 0-9), **Softmax activation*

## Training
The model is compiled using:
python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

It is trained for *10 epochs* using a *batch size of 128*:
python
history = model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1)


## Evaluation
The model is evaluated on the test set:
python
score = model.evaluate(X_test, Y_test)
print("Test accuracy:", score[1])


## Results
The final test accuracy achieved is *~98.4%*.

## Predictions
After training, the model predicts labels for test images:
python
results = model.predict(test_data)
results = np.argmax(results, axis=1)

The predictions are saved in submission.csv.

## Visualizing Results
- Randomly selected *correctly classified images*
- Randomly selected *incorrectly classified images*
python
plt.imshow(X_test[some_index].reshape(28,28), cmap='gray')
plt.title("Predicted: {}, Actual: {}".format(predicted_classes[some_index], y_test[some_index]))


## Future Improvements
- Use *Convolutional Neural Networks (CNNs)* for better accuracy.
- Experiment with *different optimizers* and *learning rates*.
- Apply *data augmentation* to improve generalization.

## References
- MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- Keras Documentation: [https://keras.io/](https://keras.io/)
- Blog Post: [https://medium.com/analytics-vidhya/get-started-with-your-first-deep-learning-project-7d989cb13ae5](https://medium.com/analytics-vidhya/get-started-with-your-first-deep-learning-project-7d989cb13ae5)

## Author
- KHUSHI KATHAK
- [https://github.com/khushi557/202401100300136_digitdetection](https://github.com/your-username)
