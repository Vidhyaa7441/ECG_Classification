import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
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

# ECG Image Processing Class
class ECG:
    def __init__(self):
        pass

    def getImage(self, uploaded_file):
        """ Reads the uploaded ECG image """
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img

    def GrayImage(self, image):
        """ Converts the ECG image to grayscale """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def DividingLeads(self, image):
        """ Simulates dividing ECG leads (Placeholder) """
        return image  # Implement actual logic if needed

    def PreprocessingLeads(self, leads):
        """ Preprocesses ECG leads (Placeholder) """
        return leads  # Implement actual logic if needed

    def SignalExtraction_Scaling(self, leads):
        """ Extracts and scales ECG signals (Placeholder) """
        return leads  # Implement actual logic if needed

    def CombineConvert1Dsignal(self):
        """ Converts ECG signals to a 1D array (Placeholder implementation) """
        return np.random.rand(100)

    def DimensionalReduction(self, signal):
        """ Performs dimensionality reduction (Placeholder implementation) """
        return signal

# Function to load images and convert them into time-series format
def load_ecg_images():
    X, y = [], []
    
    for label, category in enumerate(categories):
        folder_path = os.path.join(base_path, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            # Load image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
            img = img / 255.0  # Normalize (0 to 1)

            # Reshape image into (timesteps, features) format
            img_reshaped = img.reshape((TIMESTEPS, -1))  # (128, 128)
            
            X.append(img_reshaped)
            y.append(label)
    
    return np.array(X), np.array(y)

# Load dataset
X, y = load_ecg_images()

# One-hot encode labels
y = to_categorical(y, num_classes=len(categories))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, IMG_SIZE)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(categories), activation='softmax')  # Multi-class classification
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
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

# Save model
model.save("lstm_ecg_final_model.h5")

# Plot training accuracy & loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
