import numpy as np
import cv2
import mahotas
import pandas as pd
from scipy.ndimage import label
import os
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to read the CSV file and convert it to a list
def csv_to_list(file_path):
    with open(file_path, 'r') as file:
        # Read all lines in the file
        lines = file.readlines()
        # Convert each line to an integer and store in a list
        data_list = [int(line.strip()) for line in lines]
    return data_list

def compute_zernike_moments(image, degree):
    """Compute Zernike moments for a binary image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        (x, y), radius = cv2.minEnclosingCircle(contours[0])
        radius = int(radius)
        zm = mahotas.features.zernike_moments(image, radius, degree=degree)
        return zm
    else:
        return None

def compute_distance(zm1, zm2):
    """Compute the Euclidean distance between two Zernike moments."""
    return np.linalg.norm(np.array(zm1) - np.array(zm2))


def classify_object(sample_zernike, image, degree, threshold):
    zm = compute_zernike_moments(image, degree)
    if zm is not None:
        distance = compute_distance(sample_zernike, zm)
        if distance < threshold:
            return 1  # Focused RBC
        elif distance < threshold * 2:
            return 2  # Unfocused RBC
    return 0  # Other


def evaluate_model(sample_masks, images, true_labels, degrees, thresholds):
    results = []
    for degree in degrees:
        for mask_path in sample_masks:
            sample_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            sample_name = os.path.basename(mask_path).split('.')[0]

            sample_zernike = compute_zernike_moments(sample_mask, degree)

            for threshold in thresholds:
                predicted_labels = []
                for idx, image in enumerate(images):
                    predicted_label = classify_object(sample_zernike, image, degree, threshold)
                    predicted_labels.append(predicted_label)

                    # # Print predicted labels for Degree: 6, Threshold: 0.15
                    # if degree == 7 and threshold == 0.17 and sample_name == "20210806_FR008_01_selected_frames_00181351_object_1":
                    #     print(f'Sample Mask: {sample_name}, Image Index: {idx+1}, True Label: {true_labels[idx]}, Predicted Label: {predicted_label}')

                # Convert true labels to binary for focused RBC evaluation
                true_labels_binary = [1 if lbl == 1 else 0 for lbl in true_labels]
                predicted_labels_binary = [1 if lbl == 1 else 0 for lbl in predicted_labels]

                # Compute precision, recall, and F1-score
                precision = precision_score(true_labels_binary, predicted_labels_binary, zero_division=1)
                recall = recall_score(true_labels_binary, predicted_labels_binary, zero_division=1)
                f1 = f1_score(true_labels_binary, predicted_labels_binary, zero_division=1)

                # if degree == 7 and threshold == 0.17 and sample_name == "20210806_FR008_01_selected_frames_00181351_object_1":
                #     print(f"Sample: {sample_name} Degree: {degree}, Threshold: {threshold}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

                # if degree == 7 and threshold == 0.12 and sample_name == "20210806_FR008_01_selected_frames_00734476_object_1":
                #     print(f"Sample: {sample_name} Degree: {degree}, Threshold: {threshold}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


                results.append((sample_name, degree, threshold, precision, recall, f1))
    return results

# Load labels
labels_df = pd.read_csv('Zernike Moments/Labels_images_unfocusedRBC.csv', header=None, names=['image', 'label'])


# Specify the path to your CSV file
file_path = 'Zernike Moments/labels_better.csv'

# Convert the CSV file to a list
true_labels = csv_to_list(file_path)

# Load images
image_folder = 'Zernike Moments/separated_objects'
binary_images = [cv2.imread(f'{image_folder}/{id}.png', cv2.IMREAD_GRAYSCALE) for id in range(1,204)]

# Parameters for the experiment
degrees = [6, 7, 8, 9, 10, 11]
thresholds = [0.12, 0.15, 0.17, 0.2]

# Load sample masks
mask_folder = 'Zernike Moments/sample_masks/'
sample_masks = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.png')]

# Run the evaluation
results = evaluate_model(sample_masks, binary_images, true_labels, degrees, thresholds)

# Specify the filename
filename = "results_separated.csv"

# Write the results to the CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Sample Name", "Degree", "Threshold", "Precision", "Recall", "F1-Score"])
    # Write the data
    for sample_name, degree, threshold, precision, recall, f1 in results:
        writer.writerow([sample_name, degree, threshold, precision, recall, f1])

print(f"Results have been exported to {filename}")
