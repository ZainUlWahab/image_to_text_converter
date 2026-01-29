"""
Image Preprocessing Module
This module handles image preprocessing to improve OCR accuracy.
"""

import cv2
import numpy as np
from PIL import Image


def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image in BGR format
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        Exception: If the image cannot be loaded
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return image


def convert_to_grayscale(image):
    """
    Convert image to grayscale.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
    Returns:
        numpy.ndarray: Grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_denoising(image):
    """
    Apply denoising to reduce image noise.
    
    Args:
        image (numpy.ndarray): Grayscale image
        
    Returns:
        numpy.ndarray: Denoised image
    """
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def apply_thresholding(image):
    """
    Apply adaptive thresholding to create binary image.
    
    Args:
        image (numpy.ndarray): Grayscale image
        
    Returns:
        numpy.ndarray: Binary thresholded image
    """
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )


def deskew_image(image):
    """
    Detect and correct image skew.
    
    Args:
        image (numpy.ndarray): Binary image
        
    Returns:
        numpy.ndarray: Deskewed image
    """
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate image if skew is significant
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    
    return image


def enhance_contrast(image):
    """
    Enhance image contrast using CLAHE.
    
    Args:
        image (numpy.ndarray): Grayscale image
        
    Returns:
        numpy.ndarray: Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def resize_image(image, scale_factor=2.0):
    """
    Resize image to improve OCR accuracy.
    
    Args:
        image (numpy.ndarray): Input image
        scale_factor (float): Factor to scale the image (default: 2.0)
        
    Returns:
        numpy.ndarray: Resized image
    """
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def preprocess_image(image_path):
    """
    Complete preprocessing pipeline for an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: Preprocessed image ready for OCR
        
    Raises:
        Exception: If preprocessing fails
    """
    try:
        # Load image
        image = load_image(image_path)
        
        # Convert to grayscale
        gray = convert_to_grayscale(image)
        
        # Enhance contrast
        enhanced = enhance_contrast(gray)
        
        # Apply denoising
        denoised = apply_denoising(enhanced)
        
        # Apply thresholding
        threshold = apply_thresholding(denoised)
        
        # Deskew image
        deskewed = deskew_image(threshold)
        
        # Resize for better OCR
        resized = resize_image(deskewed, scale_factor=2.0)
        
        # Convert to PIL Image for Tesseract
        pil_image = Image.fromarray(resized)
        
        return pil_image
        
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")
