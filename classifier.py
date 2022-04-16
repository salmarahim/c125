from site import venv
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

#fetch data using openml
#here we are using the fetch_openml function to get data, the name of the data set
#  is "mnist_784,version =1"
# and getting the x and y value for it
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

# now we need to scale the data to make sure that the data points in X and Y are equal so we'll divide them
# using 255 which is the maximum pixel of the image.
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#after scaling the data we have to fit it inside our model so that it can give output with maximum accuracy.
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

# now , we have our classifier ready. Using this classifier,if we have an image, we can predict
#  the digit in the image

# if we remember what we did earlier, we were using cv2 and using cv2, we were using our devices's camera and 
# capturing each frame. Now, each frame was an image where we were doing some processing and then predicting 
# the value from it. Let's create a function to do that. 
# We'll call it get_prediction which will take the image as the parameter and make a prediction. This function will 
# take the image and convert it into a scalar quantity and then make it to gray so that the colors dont affect the
# prediction. Then we resize it into 28 by 28 scales. Then using the percentile function get the minimum pixel
# and then using the clip function give each image a number. Then using the maximum pixel we make and array.Create 
# a test sample of it and make predictions based on the sample. Fimally return the 
# test prediction.

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]