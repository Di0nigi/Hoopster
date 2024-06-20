import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Multiply
from sklearn.model_selection import train_test_split

def preprocess_video(video_path, roi=None, n_frames=5):
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

X, y = [], []

directories = [('DataModel\Hit', 1), ('DataModel\Miss', 0)]
for directory, label in directories:
    for video_file in os.listdir(directory):
        video_data = preprocess_video(os.path.join(directory, video_file))
        if video_data is not None:
            X.append(video_data)
            y.append(label)

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)

inputs = Input(shape=X_train[0].shape)

x = Conv3D(64, kernel_size=(2, 3, 3), activation='relu')(inputs)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)

attention_weights = Dense(1, activation='softmax', name='attention_weights')(x)
x = Multiply()([x, attention_weights])
x = tf.reduce_sum(x, axis=1)
x = Flatten()(x)


x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
model.save("my_trained_model_without_pretraining2.h5")
print("Accuracy:", accuracy)