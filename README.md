# Handwritten_digit_recognition-real-time-

This project is focused on developing a machine learning model to classify handwritten digits. The goal is to accurately identify the digits based on images of handwritten digits.

Dataset

The dataset used for this project is the MNIST dataset. The MNIST dataset is a collection of 70,000 images of handwritten digits, with 60,000 images for training and 10,000 images for testing. The images are 28x28 pixels and are grayscale.

Preprocessing

Before building the machine learning model, the images need to be preprocessed. The preprocessing steps include:

Reshaping the images to a flat array of 784 pixels.
Normalizing the pixel values to be between 0 and 1.
One-hot encoding the target labels.
Machine Learning Model
The machine learning model used for this project is a convolutional neural network (CNN). The CNN is a type of deep learning model that is particularly well-suited for image classification tasks.

The CNN used for this project has the following architecture:

Input layer
Convolutional layer with 32 filters and a kernel size of 3x3
ReLU activation function
Convolutional layer with 64 filters and a kernel size of 3x3
ReLU activation function
Flatten layer
Output layer with 10 neurons and a softmax activation function
The model is compiled using the categorical cross-entropy loss function and the Adam optimizer.

Training and Evaluation
The model is trained on the MNIST training dataset for 5 epochs. The training accuracy and loss are monitored during the training process. The model is then evaluated on the MNIST test dataset. The test accuracy and loss are used to assess the performance of the model.

Conclusion
The CNN model achieves a test accuracy of over 98% on the MNIST dataset, demonstrating its effectiveness in classifying handwritten digits. This model can be further improved and used for various applications, such as digit recognition in postal services, financial institutions, and many others.

Real time image capturing
Open cv is used to capture real time image.

Steps involved in extracting image captured from the camera:
  1. Specifying the required threshold value for capturing the digits in the captured image.
  2. Contouring over the digits .
  3. Bounding rectangles.
  4. Extracting those rectangles from the captured image.
  5. Using these extracted image for image classification after suitable image preprocessing.
  
  
  
