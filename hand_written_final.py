import cv2
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = mnist.load_data()


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()


## Declare the layers
layer_1 = Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1))
layer_2 = Conv2D(64, kernel_size=3, activation='relu')
layer_3 = Flatten()
layer_4 = Dense(10, activation='softmax')


## Add the layers to the model
model.add(layer_1)
model.add(layer_2)
model.add(layer_3)
model.add(layer_4)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)


def input_img():
    ima = input("Enter the path of image")
    img = cv2.imread(ima)
    return img


def capture_img():
        key = cv2. waitKey(1)
        webcam = cv2.VideoCapture(0)
        while True:
            try:
                check, frame = webcam.read()
                print(check) #prints true as long as the webcam is running
                print(frame) #prints matrix values of each framecd
                cv2.imshow("Capturing", frame)
                key = cv2.waitKey(1)
                if key == ord('s'):
                    cv2.imwrite(filename='saved_img.jpg', img=frame)
                    webcam.release()
                    img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                    img_new = cv2.imshow("Captured Image", img_new)
                    cv2.waitKey(1650)
                    cv2.destroyAllWindows()
                    print("Processing image...")
                    img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                    print("Converting RGB image to grayscale...")
                    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                    # print("Converted RGB image to grayscale...")
                    # print("Resizing image to 28x28 scale...")
                    # img_ = cv2.resize(gray,(28,28))
                    # print("Resized...")
                    img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
                    print("Image saved!")
                    break
                elif key == ord('q'):
                    print("Turning off camera.")
                    webcam.release()
                    print("Camera off.")
                    print("Program ended.")
                    cv2.destroyAllWindows()
                    break
            except(KeyboardInterrupt):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break


        img = cv2.imread('saved_img-final.jpg')
        return img




a= int(input("Do you want to input real time or saved img? (0: Real time , 1: saved img)"))


if a==0:
     img = capture_img()
   
if a==1:
     img = input_img()
   


plt.imshow(img, cmap="gray")
plt.show()




def pred(img):
    results = []
    softmaxa = []
    image = img
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
       
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
       
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]
       
        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))
       
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
       
        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)


    print("\n\n\n----------------Contoured Image--------------------")
    plt.imshow(image, cmap="gray")
    plt.show()
       
    inp = np.array(preprocessed_digits)


    for digit in preprocessed_digits:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))  
        results.append(np.argmax(prediction))
        softmaxa.append(prediction)
    return results,softmaxa
   
res,softmaxa = pred(img)




print(f"The predicted values are {res}")
print(f"Values of the model obtained are {softmaxa}")
plt.imshow(img, cmap="gray")
plt.show()
