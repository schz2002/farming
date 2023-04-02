import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define the paths to the image files
healthy_path = "healthy_plants"
diseased_path = "diseased_plants"

# Load the images and create the data set
data = []
labels = []
diseases = []

for healthy_img in os.listdir(healthy_path):
    img_path = os.path.join(healthy_path, healthy_img)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (150,150))
    data.append(img_gray)
    labels.append(0)
    diseases.append(None)

for diseased_img in os.listdir(diseased_path):
    img_path = os.path.join(diseased_path, diseased_img)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (150,150))
    data.append(img_gray)
    labels.append(1)

    # Apply image processing techniques to detect the disease
    # Here, we use thresholding, edge detection, contour analysis, and color analysis
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    perimeter = cv2.arcLength(biggest_contour, True)
    approx = cv2.approxPolyDP(biggest_contour, 0.02*perimeter, True)
    x, y, w, h = cv2.boundingRect(approx)
    if w*h > 5000: # if the bounding box area is large enough, classify as a disease
        diseases.append("bacterial spot")
    else:
        # Use color analysis to detect other diseases
        mean_color = cv2.mean(img_hsv, mask=mask)
        if mean_color[0] > 30 and mean_color[0] < 70 and mean_color[1] > 80:
            diseases.append("late blight")
        elif mean_color[0] > 90 and mean_color[0] < 140 and mean_color[1] > 150:
            diseases.append("yellow leaf curl virus")
        else:
            diseases.append("unknown")

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels, train_diseases, test_diseases = train_test_split(data, labels, diseases, test_size=0.2)

# Train the classifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(train_data.reshape(len(train_data), -1), train_labels)

# Test the classifier and print the accuracy
accuracy = classifier.score(test_data.reshape(len(test_data), -1), test_labels)
print("Accuracy:", accuracy)

# Use the classifier to predict the diseases of the test images
predictions = classifier.predict(test_data.reshape(len(test_data), -1))

# Print the actual and predicted labels and diseases
for i in range(len(predictions)):
    print("Image:", i+1)
    print("Actual Label:", test_labels[i])
    print("Predicted Label:", predictions[i])
    print("Actual Disease:", test_diseases[i])
    print("Predicted Disease:", diseases[predictions[i]])
    print()
