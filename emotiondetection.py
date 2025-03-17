import cv2
import numpy as np
from deepface import DeepFace
from IPython.display import display, clear_output
from PIL import Image
import io
from google.colab import files
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import pandas as pd
import time

def upload_video():

    print("Please upload your video file.")
    uploaded = files.upload()
    video_path = list(uploaded.keys())[0]
    return video_path

def preprocess_frame(frame):

    # Resize frame for faster processing while maintaining aspect ratio
    height, width = frame.shape[:2]
    max_dimension = 720
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # Enhance contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced

def detect_emotions(video_path):

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_emotions = []
    frame_count = 0

    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Processing video frames...")
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        # Preprocess frame
        processed_frame = preprocess_frame(frame)

        try:
            # Detect faces using cascade classifier first for speed
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            frame_emotion = []

            for (x, y, w, h) in faces:
                try:
                    # Extract face region with margin
                    margin = 40
                    y1 = max(0, y - margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    x1 = max(0, x - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    face_region = processed_frame[y1:y2, x1:x2]

                    # Analyze emotions using DeepFace
                    analysis = DeepFace.analyze(
                        face_region,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )

                    if isinstance(analysis, list):
                        analysis = analysis[0]

                    emotion_data = analysis['emotion']
                    dominant_emotion = max(emotion_data.items(), key=lambda x: x[1])

                    # Only include emotions with confidence above threshold
                    if dominant_emotion[1] > 20:  # 20% confidence threshold
                        frame_emotion.append({
                            'emotion': dominant_emotion[0],
                            'confidence': dominant_emotion[1],
                            'position': (x, y, w, h)
                        })

                        # Draw on frame
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        text = f"{dominant_emotion[0]}: {dominant_emotion[1]:.1f}%"
                        cv2.putText(frame, text, (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                except Exception as e:
                    continue

            frame_emotions.append(frame_emotion)

            # Display progress and processed frame
            if frame_count % 30 == 0:  # Update display every 30 frames
                clear_output(wait=True)
                cv2_imshow(frame)
                print(f"Processed {frame_count} frames... Time elapsed: {time.time() - start_time:.1f}s")

        except Exception as e:
            print(f"Error processing frame {frame_count}: {str(e)}")
            continue

    cap.release()

    # Calculate statistics
    all_emotions = [emotion['emotion'] for frame in frame_emotions for emotion in frame]

    if not all_emotions:
        return {
            'dominant_emotion': 'No emotions detected',
            'emotion_percentages': {},
            'frame_by_frame': frame_emotions
        }

    emotion_counts = pd.Series(all_emotions).value_counts()
    total_detections = len(all_emotions)

    emotion_stats = {
        'dominant_emotion': emotion_counts.index[0] if not emotion_counts.empty else 'No emotions detected',
        'emotion_percentages': {
            emotion: (count/total_detections * 100)
            for emotion, count in emotion_counts.items()
        },
        'frame_by_frame': frame_emotions
    }

    return emotion_stats

def visualize_emotion_stats(emotion_stats):

    if not emotion_stats['emotion_percentages']:
        print("No emotions detected to visualize.")
        return

    emotions = list(emotion_stats['emotion_percentages'].keys())
    percentages = list(emotion_stats['emotion_percentages'].values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(emotions, percentages)
    plt.title('Emotion Distribution in Video')
    plt.xlabel('Emotions')
    plt.ylabel('Percentage (%)')

    # Add percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def process_video_emotions():

    # Install required packages if not already installed
    try:
        import deepface
    except ImportError:
        print("Installing required packages...")
        !pip install deepface
        import deepface

    # Upload and process video
    video_path = upload_video()
    print("\nAnalyzing video for emotions...")
    results = detect_emotions(video_path)

    # Display results
    print("\nEmotion Analysis Results:")
    print(f"Dominant Emotion: {results['dominant_emotion']}")
    print("\nEmotion Percentages:")
    for emotion, percentage in results['emotion_percentages'].items():
        print(f"{emotion}: {percentage:.1f}%")

    # Visualize results
    visualize_emotion_stats(results)

    return results

# Example usage
if __name__ == "__main__":
    results = process_video_emotions()