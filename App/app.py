import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define paths for ECG images
base_path = r"C:\Users\Vidhyaa Thiyagarajan\OneDrive\Desktop\Final Project\lstm_ecg\lstm_ecg\input"
categories = ["Normal Person ECG Images", 
              "ECG Images of Patient that have abnormal heartbeat", 
              "ECG Images of Patient that have History of MI"]

# Define image size and timesteps
IMG_SIZE = 128  # Resize images to 128x128
TIMESTEPS = IMG_SIZE  # Each row of the image is a timestep

def load_ecg_images():
    X, y = [], []
    for label, category in enumerate(categories):
        folder_path = os.path.join(base_path, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
            img = img / 255.0  # Normalize
            img_reshaped = img.reshape((TIMESTEPS, -1))  # (128, 128)
            X.append(img_reshaped)
            y.append(label)
    return np.array(X), np.array(y)

# Load dataset
X, y = load_ecg_images()
y = to_categorical(y, num_classes=len(categories))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, IMG_SIZE)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(categories), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('lstm_ecg_model.h5', save_best_only=True)

# Train model
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=50, 
                    batch_size=32, 
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
model.save("lstm_ecg_final_model.h5")

# Streamlit App
st.title("ECG Classification using LSTM")
model = load_model("lstm_ecg_final_model.h5")
uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0  # Resize and normalize
    img = img.reshape(1, TIMESTEPS, IMG_SIZE)  # Reshape to match model input
    prediction = model.predict(img)
    class_label = np.argmax(prediction)
    st.image(uploaded_file, caption=f"Predicted Class: {categories[class_label]}", use_column_width=True)

# Plot training results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title("Model Accuracy")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].legend()
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_title("Model Loss")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend()
st.pyplot(fig)
