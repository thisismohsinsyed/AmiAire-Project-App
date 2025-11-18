import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_four_corners(contour):
    """Compute the top-left, top-right, bottom-right, and bottom-left points
    based on the sum and difference of the x and y coordinates."""
    contour = contour.squeeze()
    top_left = contour[np.argmin(contour.sum(axis=1))]
    bottom_right = contour[np.argmax(contour.sum(axis=1))]
    bottom_left = contour[np.argmin(np.diff(contour, axis=1))]
    top_right = contour[np.argmax(np.diff(contour, axis=1))]
    return top_left, top_right, bottom_left, bottom_right

def warp_perspective(image, contour):
    """Apply perspective transformation to extract ROI."""
    tl, tr, bl, br = get_four_corners(contour)
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped

def refine_corners(image, corners):
    """Refine the detected corners using sub-pixel accuracy."""
    win_size = 5
    zero_zone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined_corners = cv2.cornerSubPix(image, np.float32(corners), (win_size, win_size), zero_zone, criteria)
    return refined_corners

def adjust_corners(corners, margin=5):
    """Adjust the corners slightly inwards to avoid capturing border pixels."""
    centroid = np.mean(corners, axis=0)
    adjusted = corners + margin * (centroid - corners) / np.linalg.norm(corners - centroid, axis=1, keepdims=True)
    return adjusted

def extract_roi_from_image_array(image_array):
    """
    Function to accept an image array, apply adaptive thresholding, find and filter contours,
    extract ROI, and return both the image with detected contour and the ROI.
    """
    image = image_array.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours
    filtered_contours = [
        contour for contour in contours 
        if len(cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)) == 4 
        and cv2.isContourConvex(cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)) 
        and cv2.contourArea(contour) > 1000
    ]
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
    inner_contour = filtered_contours[1] if len(filtered_contours) > 1 else None

    roi = None
    if inner_contour is not None:
        corners = get_four_corners(inner_contour)
        refined_corners = refine_corners(gray, corners)
        adjusted_corners = adjust_corners(refined_corners)
        roi = warp_perspective(image, np.array([adjusted_corners], dtype=np.int32))

    # Draw contours for visualization
    if inner_contour is not None:
        cv2.drawContours(image, [inner_contour], -1, (0, 255, 0), 2)
    else:
        print("No inner square detected. My algorithm expects the Air meter sensor with the square.")
    # Convert images for return
    image_with_contour = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if roi is not None:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
 
         
    return image_with_contour, roi