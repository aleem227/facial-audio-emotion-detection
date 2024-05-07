import tkinter as tk
import cv2
import numpy as np
import pyaudio
import librosa
from keras.models import load_model
from collections import Counter
from keras.preprocessing import image
from PIL import Image, ImageTk

# Load the audio model
audio_model = load_model('audio_model.h5')

# Load the video model
video_model = load_model('ResNet50-face.h5')

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to record audio
class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []

    def start_recording(self, duration, sr=16000, channels=1):
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                   channels=channels,
                                   rate=sr,
                                   input=True,
                                   frames_per_buffer=1024)
        print("Recording...")
        self.frames = []
        for _ in range(0, int(sr * duration / 1024)):
            data = self.stream.read(1024)
            self.frames.append(np.frombuffer(data, dtype=np.float32))

    def stop_recording(self):
        print("Finished recording.")
        self.stream.stop_stream()
        self.stream.close()

    def get_recorded_audio(self):
        return np.hstack(self.frames)

# Function to preprocess audio
def preprocess_audio(audio_data, sr=16000, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.reshape((mfccs.shape[0], mfccs.shape[1], 1))  # Reshape directly
    return mfccs

# Function to detect emotions from video
def detect_emotions_from_video():
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

        predictions = video_model.predict(img_pixels)
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

    # Call detect_emotions_from_video again after 10ms
    video_label.after(10, detect_emotions_from_video)


# Function to detect emotions from audio
def detect_emotions_from_audio():
    audio_data = audio_recorder.get_recorded_audio()

    # Preprocess audio
    processed_audio = preprocess_audio(audio_data)
    processed_audio = np.transpose(processed_audio, (1, 0, 2)) # Transpose the array to get desired shape

    class_mapping = {
        '0': 'neutral',
        '1': 'happy',
        '2': 'disgust',
        '3': 'angry',
        '4': 'Sad'
    }

    # Predict using the audio model
    predictions = audio_model.predict(processed_audio)

    # Get the index of the class with the highest probability for each prediction
    predicted_labels_indices = np.argmax(predictions, axis=1)

    # Convert predicted_labels_indices to strings
    predicted_labels_indices_str = [str(index) for index in predicted_labels_indices]

    # Map the indices to actual class labels using class_mapping
    predicted_labels = [class_mapping[index] for index in predicted_labels_indices_str]

    # Get the most common prediction among all frames
    most_common_prediction = Counter(predicted_labels).most_common(1)[0][0]

    # Print the final predicted emotion
    emotion_label.config(text="Predicted emotion from audio: " + most_common_prediction)

# Function to close the application
def close_application():
    window.destroy()

# Function to start audio recording
def start_audio_recording():
    audio_recorder.start_recording(duration=3)

# Function to stop audio recording and detect emotions
def stop_audio_recording():
    audio_recorder.stop_recording()
    detect_emotions_from_audio()

# Create the main window
window = tk.Tk()
window.title("Real-Time Emotion Detection")
window.geometry("800x600")

# Create a label to display the video stream
video_label = tk.Label(window)
video_label.pack()

# Create a label to display the emotion predicted from audio
emotion_label = tk.Label(window)
emotion_label.pack()

# Create buttons to start and stop audio recording
start_button = tk.Button(window, text="Start Recording", command=start_audio_recording)
start_button.pack()

stop_button = tk.Button(window, text="Stop Recording and Predict", command=stop_audio_recording)
stop_button.pack()

# Create a button to close the application
close_button = tk.Button(window, text="Close", command=close_application)
close_button.pack()

# Open the video capture
cap = cv2.VideoCapture(0)

# Create an instance of AudioRecorder
audio_recorder = AudioRecorder()

# Start the emotion detection processes
detect_emotions_from_video()

# Run the Tkinter event loop
window.mainloop()

# Release the video capture and destroy the window
cap.release()
cv2.destroyAllWindows()
