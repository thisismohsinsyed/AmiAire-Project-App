import cv2
import numpy as np
import base64
import json
import os

from skimage import exposure, filters, measure, img_as_float, img_as_ubyte
from skimage.util import img_as_ubyte

def convert_to_grayscale_8bit_inmemory(image_bgr: np.ndarray, target_size: tuple = (1000, 1000)) -> np.ndarray:
    """
    Convert image to grayscale and resize it.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("Input must be a valid numpy array representing an image.")

    if image_bgr.ndim == 3:
        if image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif image_bgr.ndim == 2:
        gray = image_bgr
    else:
        raise ValueError("Unsupported image format!")

    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_CUBIC)
    return resized

def improve_background_inmemory(gray_image: np.ndarray, kernel_size: tuple = (21, 21), sigma: float = 10.0) -> np.ndarray:
    """
    Improve background by GaussianBlur-based subtraction.
    """
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("Input must be a valid 2D (grayscale) numpy array.")

    blurred = cv2.GaussianBlur(gray_image, kernel_size, sigmaX=sigma)
    float_orig = gray_image.astype(np.float32)
    float_blur = blurred.astype(np.float32)
    subtracted = float_orig - float_blur

    result_norm = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
    return result_norm.astype(np.uint8)

def rescale_intensity_inmemory(gray_image: np.ndarray, in_range_percent=(0, 20)) -> np.ndarray:
    """
    Rescale intensity based on percentile.
    """
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("Must be a valid grayscale image.")

    img_float = gray_image.astype(np.float32) / 255.0
    p_low, p_high = np.percentile(img_float, in_range_percent)
    rescaled = exposure.rescale_intensity(img_float, in_range=(p_low, p_high))
    return img_as_ubyte(rescaled)

def clahe_skimage_inmemory(gray_image: np.ndarray, clip_limit: float = 0.004, nbins: int = 12) -> np.ndarray:
    """
    Apply CLAHE enhancement.
    """
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("Must be a valid grayscale image.")

    img_float = gray_image.astype(np.float32) / 255.0
    clahe_img = exposure.equalize_adapthist(img_float, clip_limit=clip_limit, nbins=nbins)
    return img_as_ubyte(clahe_img)

def apply_sauvola_threshold_inmemory(gray_image: np.ndarray, window_size: int = 21, k: float = 0.18, invert: bool = False) -> np.ndarray:
    """
    Apply Sauvola thresholding.
    """
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("Must be a valid grayscale image.")

    img_float = gray_image.astype(np.float32) / 255.0
    thresh = filters.threshold_sauvola(img_float, window_size=window_size, k=k)
    binary = (img_float > thresh).astype(np.uint8) * 255

    if invert:
        binary = cv2.bitwise_not(binary)

    return binary

def analyze_particles_inmemory(binary_image: np.ndarray, original_bgr: np.ndarray = None, filter_params: dict = None):
    """
    Analyze particles in the binary image, filter them, and compute statistics.
    """
    if binary_image is None or binary_image.ndim != 2:
        raise ValueError("binary_image must be a 2D grayscale binary.")

    if filter_params is None:
        filter_params = {
            'min_area': 0.00,
            'max_area': 300,
            'min_solidity': 0.3,
            'max_solidity': 1.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 4.0,
            'min_feret': 0.0,
            'max_feret': 50.0
        }

    label_image = measure.label(binary_image > 0)
    regions = measure.regionprops(label_image)

    filtered_regions = []
    for region in regions:
        area = region.area
        solidity = region.solidity if hasattr(region, 'solidity') else 0.0
        if region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
        else:
            aspect_ratio = region.major_axis_length

        feret_diameter = getattr(region, 'feret_diameter_max', 0.0)

        if not (filter_params['min_area'] <= area <= filter_params['max_area']):
            continue
        if not (filter_params['min_solidity'] <= solidity <= filter_params['max_solidity']):
            continue
        if not (filter_params['min_aspect_ratio'] <= aspect_ratio <= filter_params['max_aspect_ratio']):
            continue
        if not (filter_params['min_feret'] <= feret_diameter <= filter_params['max_feret']):
            continue

        filtered_regions.append(region)

    filtered_mask = np.zeros_like(binary_image, dtype=np.uint8)
    for region in filtered_regions:
        coords = region.coords
        filtered_mask[coords[:, 0], coords[:, 1]] = 255

    image_area = filtered_mask.shape[0] * filtered_mask.shape[1]
    particle_area = np.count_nonzero(filtered_mask)
    area_percentage = (particle_area / image_area) * 100 if image_area > 0 else 0

    overlay_bgr = None
    if original_bgr is not None and original_bgr.ndim == 3:
        overlay_bgr = original_bgr.copy()
        for region in filtered_regions:
            (minr, minc, maxr, maxc) = region.bbox
            cv2.rectangle(overlay_bgr, (minc, minr), (maxc, maxr), (0, 255, 0), 2)

    return {
        'num_contours': len(filtered_regions),
        'area_percentage': area_percentage,
        'total_area': particle_area,
        'filtered_mask': filtered_mask,
        'overlay_bgr': overlay_bgr
    }

def polution_level_inmemory(analysis_results: dict, papersensor_size: tuple = (0.06, 0.06), particle_diameter: float = 10, particle_density: float = 1.65):
    """
    Calculate pollution level based on particle analysis results using the new methodology.
    """
    particle_diameter = particle_diameter * 1e-6  # Convert to meters
    particle_density = particle_density * 1e12   # Convert to kg/m^3

    area_sensor = papersensor_size[0] * papersensor_size[1]  # Area in m^2
    area_particle = np.pi * (particle_diameter/2)**2  # Area of one particle
    volume_sensor = papersensor_size[0]**3  # Volume assumed as cube for simplicity
    volume_particle = (4/3)*np.pi*(particle_diameter/2)**3  # Volume of one particle

    particles = (area_sensor * (analysis_results['area_percentage']/100)) / area_particle
    particles_per_contour = particles / analysis_results['num_contours'] if analysis_results['num_contours'] > 0 else 0
    concentration_sensor = (particles * volume_particle * particle_density) / volume_sensor

    with open(os.path.join(os.path.dirname(__file__), 'regression_params.json'), 'r') as f:
        params = json.load(f)
    
    slope = params['slope']
    intercept = params['intercept']

    concentration_standard = intercept + slope * concentration_sensor

    return {
        'num_particles': particles,
        'concentration_sensor': concentration_sensor,
        'particles_per_contour': particles_per_contour,
        'concentration_standard': concentration_standard,
    }

def classification_inmemory(concentration: float) -> str:
    """
    Classify pollution level based on concentration.
    """
    if concentration <= 10:
        return "Nivel de polución Muy bueno, menos de 10 μg/m³"
    elif concentration < 20:
        return "Nivel de polución Bueno, entre 10 to 19 μg/m³"
    elif concentration < 50:
        return "Nivel de polución Moderado, entre 20 to 49 ug/m^3"
    elif concentration < 100:
        return "Nivel de polución Malo, entre 50 to 99 μg/m³"
    elif concentration < 150:
        return "Nivel de polución Muy Malo, entre 100 to 150 μg/m³"
    else:
        return "Nivel de polución Extremo, mas de 150 μg/m³"

def process_image(roi_bgr: np.ndarray):
    """
    Process the ROI image through the complete pipeline.
    """
    gray = convert_to_grayscale_8bit_inmemory(roi_bgr, (1000, 1000))
    bg_improved = improve_background_inmemory(gray, kernel_size=(21, 21), sigma=10.0)
    rescaled = rescale_intensity_inmemory(bg_improved, in_range_percent=(0, 20))
    clahe_result = clahe_skimage_inmemory(rescaled, clip_limit=0.004, nbins=12)
    binary_mask = apply_sauvola_threshold_inmemory(clahe_result, window_size=21, k=0.18, invert=True)
    resized_bgr = cv2.resize(roi_bgr, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    analysis_results = analyze_particles_inmemory(binary_mask, resized_bgr)
    pollution_data = polution_level_inmemory(analysis_results)
    classification_str = classification_inmemory(pollution_data['concentration_standard'])

    _, mask_encoded = cv2.imencode('.png', analysis_results['filtered_mask'])
    binary_mask_b64 = base64.b64encode(mask_encoded).decode('utf-8')

    overlay_b64 = ""
    if analysis_results['overlay_bgr'] is not None:
        _, overlay_encoded = cv2.imencode('.png', analysis_results['overlay_bgr'])
        overlay_b64 = base64.b64encode(overlay_encoded).decode('utf-8')

    return {
        "analysis_results": analysis_results,
        "pollution_data": pollution_data,
        "classification": classification_str,
        "binary_mask_b64": binary_mask_b64,
        "overlay_b64": overlay_b64
    }