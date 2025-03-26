import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from keras.utils import plot_model
from keras.models import model_from_json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def classify_face_emotion(image_path):
    """
    Classifies facial emotion in an image and displays the processed image with predictions.

    Parameters:
    - image_path (str): Path to the input image.
    """
    # Load the classifier and model (hardcoded paths)
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

# Load weights and them to model
    model.load_weights('model.h5')

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Could not read image at path: {image_path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces_detected = classifier.detectMultiScale(gray_img, 1.18, 5)

    if len(faces_detected) == 0:
        print("No faces detected.")
        return  # Return early if no faces are found

    # Iterate through detected faces
    for (x, y, w, h) in faces_detected:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Region of interest (ROI) for the face
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        # Preprocess the face image for model prediction
        img_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        # Predict emotion
        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))

        # Emotion labels (modify if different labels were used during training)
        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]

        # Add predicted emotion text to the image
        cv2.putText(img, predicted_emotion, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Resize the image for better viewing (optional)
    resized_img = cv2.resize(img, (1024, 768))

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Facial Emotion Recognition")
    plt.show()

# Any image u want to select:

# image_path = 'angry.jpeg'
# classify_face_emotion(image_path)

# image_path = 'disgust.jpeg'
# classify_face_emotion(image_path)

# image_path = 'disgust2.jpeg'
# classify_face_emotion(image_path)

# image_path = 'happy.jpeg'
# classify_face_emotion(image_path) 

# image_path = 'happy2.jpeg'
# classify_face_emotion(image_path)

# image_path = 'neutral.jpeg'
# classify_face_emotion(image_path)

# image_path = 'sad.jpeg'
# classify_face_emotion(image_path)

# image_path = 'sad2.jpg'
# classify_face_emotion(image_path)

# image_path = 'surprised.jpeg'
# classify_face_emotion(image_path)

# image_path = 'fear.jpeg'
# classify_face_emotion(image_path)