import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from tqdm.notebook import tqdm
import csv

# Function to create a dataframe for labeled datasets
def create_dataframe(ai_dir, real_dir):
    image_paths = []
    labels = []

    # AI images
    for image_name in os.listdir(ai_dir):
        image_paths.append(os.path.join(ai_dir, image_name))
        labels.append('AI')

    # Real images
    for image_name in os.listdir(real_dir):
        image_paths.append(os.path.join(real_dir, image_name))
        labels.append('Real')

    return pd.DataFrame({'image': image_paths, 'label': labels})

# Function to create a dataframe for unlabeled test datasets
def create_test_dataframe(test_dir):
    image_paths = []
    for image_name in os.listdir(test_dir):
        if image_name != "image_62.jpg":  # Skip problematic image
            image_paths.append(os.path.join(test_dir, image_name))
    return pd.DataFrame({'image': image_paths})

# Function to extract features from images
def extract_features(images, target_size=(128, 128)):
    features = []
    for image in tqdm(images, desc="Extracting features"):
        img = load_img(image, target_size=target_size)  # Resize images
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    return features.reshape(features.shape[0], target_size[0], target_size[1], 3)

# Paths to datasets
TRAIN_AI_DIR = "/kaggle/input/cynaptics-training-ai"
TRAIN_REAL_DIR = "/kaggle/input/cynaptics-training-real"
TEST_DIR = "/kaggle/input/cynaptics-testing-dataset"

# Create dataframes
train_df = create_dataframe(TRAIN_AI_DIR, TRAIN_REAL_DIR)
test_df = create_test_dataframe(TEST_DIR)

# Extract features and normalize
x_train = extract_features(train_df['image']) / 255.0  # Normalize pixel values
x_test = extract_features(test_df['image']) / 255.0

# Encode labels
le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df['label'])  # Encode 'AI' -> 0, 'Real' -> 1
y_train = to_categorical(train_df['label_encoded'], num_classes=2)  # One-hot encode labels

# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 output classes: 'AI' and 'Real'
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the best model weights
weights_file = "best_model.keras"
checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True, mode='min')

# Train the model
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,  # Use 20% of the data for validation
    callbacks=[checkpoint]
)

# Load the best weights
model.load_weights(weights_file)

# Predict on test data
predictions = model.predict(x_test)
predicted_labels = le.inverse_transform(np.argmax(predictions, axis=1))  # Decode predictions

# Prepare the output dataframe
test_df['Label'] = predicted_labels  # Add the predicted labels to the test dataframe

# Extract IDs (image filenames without extension) for the CSV file
test_df['Id'] = test_df['image'].apply(lambda x: os.path.basename(x).replace('.jpg', ''))  # Remove .jpg extension

# Sort the dataframe by numeric ID for consistent output order
test_df['Numeric_Id'] = test_df['Id'].str.extract('(\d+)').astype(int)  # Extract numeric part of the ID
test_df = test_df.sort_values(by='Numeric_Id').reset_index(drop=True)

# Reformat the ID to include "image_" prefix
test_df['Id'] = 'image_' + test_df['Numeric_Id'].astype(str)

# Create the CSV file
output_file = "predictions.csv"
test_df[['Id', 'Label']].to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
