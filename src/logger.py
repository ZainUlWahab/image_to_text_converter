"""
Logging Module
Provides comprehensive logging for OCR operations and debugging.
"""

import logging
import os
from datetime import datetime


def setup_logger(name='ocr_app', log_dir='logs'):
    """
    Set up and configure a logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_dir (str): Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (detailed logs)
    log_filename = os.path.join(log_dir, f'ocr_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simpler logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_ocr_results(logger, ocr_data):
    """
    Log OCR extraction results.
    
    Args:
        logger: Logger instance
        ocr_data (dict): OCR data containing text and boxes
    """
    logger.info("=" * 60)
    logger.info("OCR EXTRACTION RESULTS")
    logger.info("=" * 60)
    
    text = ocr_data.get('text', '')
    boxes = ocr_data.get('boxes', [])
    confidence = ocr_data.get('confidence', 0)
    
    logger.info(f"Total words detected: {len(boxes)}")
    logger.info(f"Average confidence: {confidence:.2f}%")
    logger.info(f"Extracted text length: {len(text)} characters")
    
    logger.debug("Full extracted text:")
    logger.debug("-" * 60)
    logger.debug(text)
    logger.debug("-" * 60)
    
    # Log sample boxes
    if boxes:
        logger.debug(f"Sample word boxes (first 10):")
        for i, box in enumerate(boxes[:10]):
            logger.debug(f"  {i+1}. '{box['text']}' - Confidence: {box['conf']:.1f}% - Position: ({box['left']}, {box['top']})")


def log_format_detection(logger, formatted_data):
    """
    Log formatting detection results.
    
    Args:
        logger: Logger instance
        formatted_data (dict): Formatted data with paragraphs and styling
    """
    logger.info("=" * 60)
    logger.info("FORMAT DETECTION RESULTS")
    logger.info("=" * 60)
    
    paragraphs = formatted_data.get('paragraphs', [])
    tables = formatted_data.get('tables', [])
    
    logger.info(f"Total paragraphs detected: {len(paragraphs)}")
    logger.info(f"Total tables detected: {len(tables)}")
    
    # Log paragraph details
    for i, para_lines in enumerate(paragraphs):
        logger.debug(f"\nParagraph {i+1}:")
        for j, line in enumerate(para_lines):
            formatting = []
            if line.get('is_bold'):
                formatting.append('BOLD')
            if line.get('is_italic'):
                formatting.append('ITALIC')
            if line.get('heading_level', 0) > 0:
                formatting.append(f'H{line["heading_level"]}')
            
            format_str = f"[{', '.join(formatting)}]" if formatting else ""
            logger.debug(f"  Line {j+1} {format_str}: {line['text'][:50]}...")
    
    # Log table details
    for i, table in enumerate(tables):
        rows = table.get('rows', [])
        cols = table.get('cols', 0)
        logger.debug(f"\nTable {i+1}: {len(rows)} rows x {cols} columns")


def log_processing_step(logger, step_name, details=""):
    """
    Log a processing step.
    
    Args:
        logger: Logger instance
        step_name (str): Name of the processing step
        details (str): Additional details
    """
    logger.info(f">>> {step_name}")
    if details:
        logger.info(f"    {details}")


def log_error(logger, error, context=""):
    """
    Log an error with context.
    
    Args:
        logger: Logger instance
        error (Exception): The error that occurred
        context (str): Context where the error occurred
    """
    logger.error("=" * 60)
    logger.error(f"ERROR: {context}" if context else "ERROR OCCURRED")
    logger.error(f"Type: {type(error).__name__}")
    logger.error(f"Message: {str(error)}")
    logger.error("=" * 60)


# Create global logger instance
logger = setup_logger()
