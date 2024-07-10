import cv2
import numpy as np
from teachable_machine import TeachableMachine
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the modified model
# model = TeachableMachine(model_path=r"D:\ai_vision\TeachableMachine\keras_model.h5",
#                          labels_file_path=r"D:\ai_vision\TeachableMachine\labels.txt")
model= load_model(r"D:\ai_vision\TeachableMachine\keras_model.h5", compile=False)
# Load the labels
class_names = open(r"D:\ai_vision\TeachableMachine\labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
cap = cv2.VideoCapture(0)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Draw the rectangle around the object
    start_point = (50, 50)
    end_point = (frame.shape[1] - 50, frame.shape[0] - 50)
    color = (0, 255, 0)  # Green color for the rectangle
    thickness = 2

    if confidence_score > 0.8:
        color = (0, 255, 0)  # Green for high confidence
    else:
        color = (0, 0, 255)  # Red for low confidence

    cv2.rectangle(frame, start_point, end_point, color, thickness)

    # Put text with class name and confidence score
    text = f"{class_name.strip()} : {confidence_score:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video Stream', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()