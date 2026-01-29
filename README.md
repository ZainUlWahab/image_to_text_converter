# Document Converter

## Overview
Transform images into editable Word documents with intelligent formatting preservation. Built with **LightOnOCR-1B-1025** vision-language model and dual-pipeline architecture for superior text extraction and layout analysis.

## Students
| Name | Roll Number |
|------|-------------|
| Sahal Saeed | 22i-0476 |
| Zain Ul Wahab | 22i-0491 |

## Features
- **LightOnOCR-1B-1025** - State-of-the-art OCR model
- **GPU Accelerated** - Optimized for NVIDIA GPUs
- **Smart Preprocessing** - Automatic image enhancement
- **Layout Analysis** - Detects bold, italic, and alignment
- **Markdown Support** - Tables, lists, headings, code blocks
- **Word Export** - Professional .docx generation
- **Modern UI** - Clean Streamlit interface

## Requirements
- Python 3.9+
- NVIDIA GPU with CUDA (optional, CPU fallback available)
- 4GB+ RAM

## Installation

```bash
# 1. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install transformers from source
pip install git+https://github.com/huggingface/transformers

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The app will open at http://localhost:8501

## How It Works

**Dual-Pipeline Architecture:**
1. **OCR Pipeline** - LightOnOCR extracts text in markdown format
2. **Layout Pipeline** - Computer vision analyzes visual formatting (bold, italic, alignment)
3. **Document Generation** - Combines both pipelines to create formatted Word document

## Project Structure
```
phase1/
├── src/
│   ├── ocr_engine.py        # LightOnOCR text extraction
│   ├── layout.py            # Visual layout analysis
│   ├── docx_writer.py       # Word document generation
│   ├── image_processor.py   # Image preprocessing
│   ├── streamlit_app.py     # Web interface
│   └── logger.py            # Logging system
├── output/                  # Generated documents
├── logs/                    # Application logs
├── main.py                  # Entry point
└── requirements.txt         # Dependencies
```

## Workflow

1. **Upload Image** - JPG/PNG document image
2. **Preprocessing** - Automatic enhancement (denoise, deskew, contrast)
3. **Analysis** - Dual-pipeline processing
   - Text extraction with OCR
   - Visual formatting detection
4. **Generate** - Create formatted Word document
5. **Download** - Get .docx file

## Performance

| Hardware | Speed | Accuracy |
|----------|-------|----------|
| GPU (RTX 5070) | ~2-3s | 95%+ |
| CPU | ~60-120s | 95%+ |

## Supported Features

- Paragraphs and headings  
- Tables with multiple columns  
- Bulleted and numbered lists  
- Bold and italic text  
- Text alignment (left/center/right)  
- Markdown code blocks  
- Inline formatting  

## License
MIT License

## Deployed
Link: https://zainulwahab-image-to-text-converter-srcstreamlit-app-qwj96t.streamlit.app/
