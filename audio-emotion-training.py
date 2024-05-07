import pyaudio
import numpy as np
import librosa
from keras.models import load_model
from collections import Counter

# Function to record audio
def record_audio(duration, sr=16000, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sr,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")
    frames = []
    for _ in range(0, int(sr * duration / 1024)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.hstack(frames)

# Function to preprocess audio
def preprocess_audio(audio_data, sr=16000, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.reshape((mfccs.shape[0], mfccs.shape[1], 1))  # Reshape directly
    return mfccs

# Load the model
model = load_model('best_model.h5')

# Record audio
duration = 3  # seconds
audio_data = record_audio(duration)

# Preprocess audio
processed_audio = preprocess_audio(audio_data)
processed_audio = np.transpose(processed_audio, (1, 0, 2)) # Transpose the array to get desired shape
print("Shape of preprocessed audio:", processed_audio.shape)


class_mapping = {
    '0': 'neutral',
    '1': 'happy',
    '2': 'disgust',
    '3': 'angry',
    '4': 'Sad'
                }


# Predict using the model
predictions = model.predict(processed_audio)

# Get the index of the class with the highest probability for each prediction
predicted_labels_indices = np.argmax(predictions, axis=1)

# Convert predicted_labels_indices to strings
predicted_labels_indices_str = [str(index) for index in predicted_labels_indices]

# Map the indices to actual class labels using class_mapping
predicted_labels = [class_mapping[index] for index in predicted_labels_indices_str]

# Get the most common prediction among all frames
most_common_prediction = Counter(predicted_labels).most_common(1)[0][0]

# Print the final predicted emotion
print("Predicted emotion:", most_common_prediction)