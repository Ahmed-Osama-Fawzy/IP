import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import uuid
import os

def preprocess_image(image):
    # Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)

    # Apply morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    return closed

def detect_text_regions(image):
    mser = cv2.MSER_create()

    # Detect regions
    regions, _ = mser.detectRegions(image)

    # Convert regions to bounding boxes
    bboxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
    return bboxes

def cluster_regions(bboxes, eps=13):
    # Compute centroids for clustering
    centroids = np.array([[x + w/2, y + h/2] for (x, y, w, h) in bboxes])

    # Cluster regions based on spatial proximity
    dbscan = DBSCAN(eps=eps, min_samples=2)
    labels = dbscan.fit_predict(centroids)

    # Group bounding boxes by cluster label
    clusters = {}
    for label, bbox in zip(labels, bboxes):
        if label == -1:  # noise
            continue
        clusters.setdefault(label, []).append(bbox)
    return clusters

def merge_bboxes(bboxes):
    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[0] + b[2] for b in bboxes)
    y_max = max(b[1] + b[3] for b in bboxes)
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def extract_text_from_clusters(image, eps, preprocess=True):
    image_copy = image.copy()
    if preprocess:
        image_copy = preprocess_image(image)
    bboxes = detect_text_regions(image_copy)
    clusters = cluster_regions(bboxes,eps)
    merged_bboxes = [merge_bboxes(cluster) for cluster in clusters.values()]
    for (x, y, w, h) in merged_bboxes:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0), 2)
    return image_copy

def run(img, eps):
    # Load the image
    image = cv2.imread(img)
    text_from_image = extract_text_from_clusters(image, eps, preprocess=False)
    text_from_preprocessed_image = extract_text_from_clusters(image, eps, preprocess=True)

    base = os.path.splitext(os.path.basename(img))[0]
    out1_name = f"{base}_output1_{uuid.uuid4().hex[:6]}.jpg"
    out2_name = f"{base}_output2_{uuid.uuid4().hex[:6]}.jpg"
    
    out1_path = os.path.join("static/uploads", out1_name)
    out2_path = os.path.join("static/uploads", out2_name)
    
    cv2.imwrite(out1_path, text_from_image)
    cv2.imwrite(out2_path, text_from_preprocessed_image)
    
    return out1_name, out2_name