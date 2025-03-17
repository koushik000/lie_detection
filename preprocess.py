import pandas as pd
import numpy as np
import cv2
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.colab import files

# Step 1: Upload the video and preprocessed CSV file from your local machine
print("Please upload your video file.")
uploaded_video = files.upload()
fpath = list(uploaded_video.keys())[0]

print("Please upload the corresponding preprocessed CSV file.")
uploaded_csv = files.upload()
csv_path = list(uploaded_csv.keys())[0]

# Step 2: Extract MFCC from audio in video
cam = cv2.VideoCapture(fpath)
fps = cam.get(cv2.CAP_PROP_FPS)
audio, sr = librosa.load(fpath)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=int(sr // fps))
array_MFCC = np.transpose(mfcc)

# Step 3: Load preprocessed Action Units data from the CSV
df = pd.read_csv(csv_path)
col_list = list(filter(lambda x: x.startswith("AU") and x.endswith("r"), df.columns))
array_AU = df[col_list].to_numpy()

# Step 4: Optionally resize to 100 frames
if array_AU.shape[0] > 100:
    hop = np.size(array_AU, 0) // 100
    array_AU = np.array([array_AU[i * hop] for i in range(100)])
if array_MFCC.shape[0] > 100:
    hop = np.size(array_MFCC, 0) // 100
    array_MFCC = np.array([array_MFCC[i * hop] for i in range(100)])

# Step 5: Ensure both feature arrays are of the same length by trimming
min_frames = min(array_AU.shape[0], array_MFCC.shape[0])
array_AU = array_AU[:min_frames]
array_MFCC = array_MFCC[:min_frames]

# Step 6: Concatenate AU and MFCC features
array = np.concatenate((array_AU, array_MFCC), axis=1)

# Step 7: Load the model and make predictions
model = load_model("/content/lstm_model.h5")  # Make sure you've uploaded the model file
x = np.expand_dims(array, axis=0)
y = model.predict(x)[0][0]

# Step 8: Output the result
print("\nResult:", "Truthful" if y >= 0.5 else "Deceptive")
print("Output score:", y)