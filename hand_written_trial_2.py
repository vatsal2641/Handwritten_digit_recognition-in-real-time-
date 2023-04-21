import cv2
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import streamlit as st

st.title("Handwritten digit recognition system")


nav = st.sidebar.radio("Navigation", ["Introduction","Prediction"])
if nav=="Introduction":
     
     st.image("handwitten digit recognition.png")
     st.write("Handwritten digit recognition using MNIST dataset is a major project made with the help of Neural Network. It basically detects the scanned images of handwritten digits.")

     st.write("Handwritten digit recognition is the process to provide the ability to machines to recognize human handwritten digits.")

if nav=="Prediction":

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    first,last = st.columns(2)
    first.write("Real time.")
    first.image("me.png")

    last.write("Input Image.")
    last.image("test_image.jpg")
    
    

    
    
    r= st.checkbox("Real time")
    i = st.checkbox("Input image")


    if r:
        
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
        img = capture_img()
    
    elif i:
        ii = st.text_input("Enter the path of the image")
        def input_img(pathi):
            ima = pathi
            img = cv2.imread(ima)
            return img
         
        img = input_img(ii)

    
    train_model = st.checkbox("Train model")

    if train_model:

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
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

        st.balloons()

    


    # # a= int(input("Do you want to input real time or saved img? (0: Real time , 1: saved img)"))
    

    # if s=="Real time":
        
        
    # if s=="Saved image": 
        
        

    # plt.imshow(img, cmap="gray")
    # plt.show()


    
    r= st.checkbox("Result")
    if r:
    
        def pred(img):
            results = []
            softmaxa = []
            image = img
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
                
                # print ("\n\n---------------------------------------\n\n")
                # print ("=========PREDICTION============ \n\n")
                # # plt.imshow(digit.reshape(28, 28), cmap="gray")
                # # plt.show()
                # print("\n\nFinal Output: {}".format(np.argmax(prediction)))
                
                # print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
                
                # hard_maxed_prediction = np.zeros(prediction.shape)
                # hard_maxed_prediction[0][np.argmax(prediction)] = 1
                # print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
                # print ("\n\n---------------------------------------\n\n")
                results.append(np.argmax(prediction))
                softmaxa.append(prediction)
                return results,softmaxa
        res,softmaxa = pred(img)

        st.write(f"Values of the model obtained are {softmaxa}")
        st.write(f"The predicted values are: ")
        for i in res:
                st.write(i)
        
        st.image(img)
        print(res)
    

# plt.imshow(img, cmap="gray")
# plt.show()

