import tkinter as tk
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageTk

# Load the pre-trained model
model = load_model('ResNet50.h5')

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to process video frames and detect emotions
def detect_emotions():
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        roi_gray_resized = cv2.resize(roi_gray, (224, 224))  # Resize the input image
        roi_gray_resized_rgb = cv2.cvtColor(roi_gray_resized, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        img_pixels = np.expand_dims(roi_gray_resized_rgb, axis=0)  # Add batch dimension
        img_pixels = img_pixels / 255.0  # Normalize

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ['Angry', 'Disgust', 'Happy', 'Neutral', 'Sad']
        predicted_emotion = emotions[max_index]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Call detect_emotions again after 10ms
    video_label.after(10, detect_emotions)

# Function to close the application
def close_application():
    window.destroy()

# Create the main window
window = tk.Tk()
window.title("Real-Time Emotion Detection")
window.geometry("800x600")

# Create a label to display the video stream
video_label = tk.Label(window)
video_label.pack()

# Create a button to close the application
close_button = tk.Button(window, text="Close", command=close_application)
close_button.place(relx=.5, rely=.8)
close_button.pack()

# Open the video capture
cap = cv2.VideoCapture(0)

# Start the emotion detection process
detect_emotions()

# Run the Tkinter event loop
window.mainloop()

# Release the video capture and destroy the window
cap.release()
cv2.destroyAllWindows()
