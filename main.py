git# main.py

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from scipy.spatial.distance import cosine  # For cosine similarity
import csv

# Initialize HandShapeFeatureExtractor
feature_extractor = HandShapeFeatureExtractor.get_instance()

# Paths to gesture videos
training_videos_path = 'traindata'  # Change to the correct path for your training videos
test_videos_path = 'test'  # Path to test videos (adjust as needed)
frames_output_path_train = 'frames_output_train'  # Directory to save extracted frames for training
frames_output_path_test = 'frames_output_test'  # Directory to save extracted frames for test

# Lists to store the feature vectors
training_feature_vectors = []
test_feature_vectors = []

# Function to extract features from videos
def extract_features_from_videos(video_folder, frames_path, feature_vector_list):
    count = 0
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)

        # Extract the middle frame of the gesture video and save as an image
        frameExtractor(video_path, frames_path, count)
        frame_file = os.path.join(frames_path, f"{count+1:05d}.png")

        # Read the frame
        frame = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)

        # Extract hand shape feature using HandShapeFeatureExtractor
        feature_vector = feature_extractor.extract_feature(frame)

        # Store the feature vector
        feature_vector_list.append(feature_vector)

        count += 1
        print(f"Processed video {count}: {video_file}")

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
print("Extracting penultimate layer for training data...")

# Extract features from all training videos
extract_features_from_videos(training_videos_path, frames_output_path_train, training_feature_vectors)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
print("Extracting penultimate layer for test data...")

# Extract features from all test videos
extract_features_from_videos(test_videos_path, frames_output_path_test, test_feature_vectors)

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
def recognize_gesture(test_vector, training_vectors):
    # Calculate cosine similarity between test vector and all training vectors
    similarities = [cosine(test_vector, train_vector) for train_vector in training_vectors]
    # Find the index of the minimum cosine distance
    recognized_gesture_idx = np.argmin(similarities)
    return recognized_gesture_idx

# List to store recognized gestures
recognized_gestures = []

# Recognize gestures for all test videos
for test_vector in test_feature_vectors:
    gesture_label = recognize_gesture(test_vector, training_feature_vectors)
    recognized_gestures.append(gesture_label)
    print(f"Recognized gesture: {gesture_label}")

# =============================================================================
# Save the results to "Results.csv"
# =============================================================================
print("Saving results to Results.csv...")

# Save recognized gestures to a CSV file
with open('Results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    for gesture in recognized_gestures:
        writer.writerow([gesture])

print("Task 3 completed successfully!")

