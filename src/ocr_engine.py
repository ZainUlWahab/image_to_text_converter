"""
OCR Engine Module
This module handles text extraction using LightOnOCR-1B-1025.
Optimized for RTX 5070 GPU with CUDA acceleration.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


class LightOnOCREngine:
    """
    Thin wrapper around LightOnOCR-1B inference using official HuggingFace approach.
    """

    def __init__(self, model_id: str = "lightonai/LightOnOCR-1B-1025") -> None:
        self.model_id = model_id

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        if self.device == "cuda":
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ Using CPU (GPU not available)")
        
        print(f"📥 Loading {self.model_id}...")

        # Use LightOnOcrProcessor as per official HuggingFace documentation
        self.processor = LightOnOcrProcessor.from_pretrained(self.model_id)
        
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype
        ).to(self.device)

        self.model.eval()
        print("✅ Model loaded successfully!")

    @torch.inference_mode()
    def ocr(self, image: Image.Image, max_new_tokens: int = 1024) -> str:
        """
        Extract text from image using official HuggingFace approach.
        """
        # Official conversation format with actual image
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        
        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Move to device with proper dtype
        inputs = {
            k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device) 
            for k, v in inputs.items()
        }
        
        # Generate
        output_ids = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        
        # Decode using processor directly
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return output_text.strip()


# Singleton instance for caching
_engine_instance = None


def get_ocr_engine(model_id: str = "lightonai/LightOnOCR-1B-1025") -> LightOnOCREngine:
    """Get or create OCR engine instance (cached)."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LightOnOCREngine(model_id=model_id)
    return _engine_instance


# Legacy API compatibility
def extract_text(image: Image.Image, max_new_tokens: int = 1024) -> str:
    """Legacy function for compatibility."""
    engine = get_ocr_engine()
    return engine.ocr(image, max_new_tokens=max_new_tokens)


def process_image_ocr(image: Image.Image) -> dict:
    """
    Legacy compatibility function.
    Returns dict with text for backward compatibility with existing code.
    """
    text = extract_text(image)
    return {
        'text': text,
        'boxes': [],  # No longer used - layout analysis is separate
        'confidence': 95.0
    }
