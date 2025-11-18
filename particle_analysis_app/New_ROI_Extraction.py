# ROI_Extraction.py
import cv2
import numpy as np

def extract_roi_from_image_array(image_bgr: np.ndarray):
    """
    Extract the region of interest (ROI) defined by a large black square
    from an in-memory BGR image (NumPy array).
    Returns (image_with_contour, roi) if successful, otherwise (None, None).
    """

    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("Input must be a valid BGR image (numpy array).")

    # --- Utility functions ---

    def order_points(pts):
        """
        Given an array of 4 corner points (x, y),
        return them in order: [top-left, top-right, bottom-right, bottom-left].
        """
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def warp_perspective(image, corners):
        """
        Warp the image to the rectangle defined by `corners`.
        `corners` can be in any order; we reorder them internally.
        """
        corners = np.array(corners, dtype="float32").reshape(-1, 2)
        if corners.shape[0] != 4:
            raise ValueError("Expected exactly 4 corner points.")

        rect = order_points(corners)  # [tl, tr, br, bl]
        (tl, tr, br, bl) = rect

        width_bottom = np.linalg.norm(br - bl)
        width_top = np.linalg.norm(tr - tl)
        maxWidth = int(max(width_bottom, width_top))

        height_right = np.linalg.norm(tr - br)
        height_left = np.linalg.norm(tl - bl)
        maxHeight = int(max(height_right, height_left))

        dst = np.array([
            [0,          0],
            [maxWidth-1, 0],
            [maxWidth-1, maxHeight-1],
            [0,          maxHeight-1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
        return warped

    def refine_corners(gray_image, corners):
        """
        Refine the corner points with cornerSubPix for better accuracy.
        """
        corners = np.array(corners, dtype=np.float32)
        win_size = (5, 5)
        zero_zone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners_reshaped = corners.reshape(-1, 1, 2)
        refined = cv2.cornerSubPix(gray_image, corners_reshaped, win_size, zero_zone, criteria)
        return refined.reshape(-1, 2)

    def adjust_corners(corners, margin=15):
        """
        Pull each corner inwards by 'margin' pixels to avoid black border.
        """
        centroid = np.mean(corners, axis=0)
        distances = np.linalg.norm(corners - centroid, axis=1, keepdims=True)
        if np.any(distances == 0):
            return corners
        adjusted = corners + margin * (centroid - corners) / distances
        return adjusted

    # --- Main ROI detection logic ---

    color_image = image_bgr.copy()
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    binary = cv2.adaptiveThreshold(
        gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 121, 2
    )

    #opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    #closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    max_image_area = h * w

    # Filter candidate squares
    candidate_contours = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 70000 and area < 0.9 * max_image_area:
                candidate_contours.append(approx)

    candidate_contours.sort(key=cv2.contourArea, reverse=True)

    similarity_threshold = 0.85
    inner_contour = None

    if len(candidate_contours) >= 2:
        chosen_pair_found = False
        for i in range(len(candidate_contours) - 1):
            area1 = cv2.contourArea(candidate_contours[i])
            area2 = cv2.contourArea(candidate_contours[i + 1])
            ratio = area2 / area1
            if ratio >= similarity_threshold:
                inner_contour = candidate_contours[i + 1]
                chosen_pair_found = True
                break
        if not chosen_pair_found:
            inner_contour = candidate_contours[0]
    elif len(candidate_contours) == 1:
        inner_contour = candidate_contours[0]
    else:
        inner_contour = None

    roi = None
    image_with_contour = None
    if inner_contour is not None:
        corners = np.squeeze(inner_contour, axis=1).astype(np.float32)
        refined_corners = refine_corners(gray_blurred, corners)
        adjusted_corners = adjust_corners(refined_corners, margin=15)
        roi = warp_perspective(color_image, adjusted_corners)

    # Create an image_with_contour for visualization
    image_with_contour = color_image.copy()
    # Mark up to 3 largest candidate contours with different colors
    # colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255)]
    # for i, cnt in enumerate(candidate_contours[:3]):
    #     cv2.drawContours(image_with_contour, [cnt], -1, colors[i], 2)

    # Highlight the chosen inner contour in green
    if inner_contour is not None:
        cv2.drawContours(image_with_contour, [inner_contour], -1, (0, 255, 0), 2)

    # If no valid ROI found, return (None, None)
    if roi is None:
        return None, None

    return image_with_contour, roi
