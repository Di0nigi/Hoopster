import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('my_trained_model_without_pretraining.h5')
def preprocess_video(video_path, n_frames=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_step = total_frames // n_frames
    frames = []
   
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frames_step)
        ret, frame = cap.read()
        if not ret:
            continue
       
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
   
    cap.release()

    if len(frames) != n_frames:
        return None

    frames = np.array(frames)
    return frames / 255.  # Normalize data




def predict(video_path):
    preprocessed_video = preprocess_video(video_path)
    if preprocessed_video is not None:
        prediction = model.predict(np.expand_dims(preprocessed_video, axis=0))

        label = "Hits" if prediction[0][0] > 0.5 else "Misses"
        print(label)
    else:
        print("Couldn't process the video properly.")



predict("DataModel\Hit\output_100.mp4")