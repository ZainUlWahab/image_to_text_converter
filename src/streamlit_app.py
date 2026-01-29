"""
Streamlit GUI Application
Modern web-based interface for OCR with formatting preservation.
Uses dual-pipeline approach: OCR for text + Layout analysis for formatting.
"""

import streamlit as st
from PIL import Image
import os
import sys
import io

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from image_processor import preprocess_image
from ocr_engine import get_ocr_engine
from layout import analyze_layout, render_layout_debug
from docx_writer import build_docx
from logger import logger, log_processing_step, log_error


# Page configuration
st.set_page_config(
    page_title="Document Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalistic UI
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main container */
    .main {
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
        background-color: #ffffff;
    }
    
    /* Block container background */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #ffffff;
    }
    
    /* Header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #4a90e2;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #4a90e2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #4a90e2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.8rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #4a90e2 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #4a90e2;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #495057;
        background-color: #f8f9fa;
        border-radius: 6px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4a90e2;
        border-radius: 8px;
        padding: 2rem;
        background-color: #f8f9fa;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #4a90e2;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    /* Subheaders */
    h3 {
        color: #495057;
        font-weight: 600;
    }
    
    /* All text elements */
    p, label, span, div {
        color: #333333;
    }
    
    /* Caption text */
    .caption {
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'preprocessed_image' not in st.session_state:
        st.session_state.preprocessed_image = None
    if 'ocr_text' not in st.session_state:
        st.session_state.ocr_text = None
    if 'layout' not in st.session_state:
        st.session_state.layout = None
    if 'docx_bytes' not in st.session_state:
        st.session_state.docx_bytes = None


def reset_session():
    """Reset all session state."""
    st.session_state.step = 1
    st.session_state.uploaded_image = None
    st.session_state.preprocessed_image = None
    st.session_state.ocr_text = None
    st.session_state.layout = None
    st.session_state.docx_bytes = None


def main():
    """Main application function."""
    initialize_session_state()
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    
    # Header
    st.markdown('<div class="main-header">Document Converter</div>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Transform images into editable Word documents with intelligent formatting</p>',
                unsafe_allow_html=True)
    
    # Main content
    current_step = st.session_state.step
    
    # Progress indicator - 5 steps now
    progress_value = (current_step - 1) / 4  # 5 steps total
    st.progress(progress_value)
    
    step_names = ["Upload Image", "Preprocessing", "OCR & Layout", "Generate Document", "Download"]
    st.caption(f"Step {current_step} of 5: {step_names[current_step - 1]}")
    
    st.markdown("---")
    
    # Step 1: Upload Image
    if current_step == 1:
        st.subheader("Step 1: Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image with English text"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.markdown("**Image Information**")
                st.write(f"Format: {image.format}")
                st.write(f"Size: {image.size[0]} × {image.size[1]} pixels")
                st.write(f"Mode: {image.mode}")
                st.write("")
                
                if st.button("Start Processing", use_container_width=True, type="primary"):
                    with st.spinner("Preprocessing image..."):
                        try:
                            # Save uploaded file temporarily
                            temp_path = os.path.join(output_dir, "temp_upload.png")
                            os.makedirs(output_dir, exist_ok=True)
                            image.save(temp_path)
                            
                            log_processing_step(logger, "Image Upload", f"Uploaded {uploaded_file.name}")
                            
                            # Preprocess
                            log_processing_step(logger, "Image Preprocessing")
                            preprocessed = preprocess_image(temp_path)
                            st.session_state.preprocessed_image = preprocessed
                            st.session_state.temp_image_path = temp_path
                            
                            # Move to preprocessing preview
                            st.session_state.step = 2
                            st.rerun()
                            
                        except Exception as e:
                            log_error(logger, e, "Image Processing")
                            st.error(f"Error processing image: {str(e)}")
    
    # Step 2: Preprocessing Preview
    elif current_step == 2:
        st.subheader("Step 2: Preprocessing Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(st.session_state.uploaded_image, use_container_width=True)
        
        with col2:
            st.markdown("**Preprocessed Image**")
            st.image(st.session_state.preprocessed_image, use_container_width=True)
        
        st.info("The image has been preprocessed with: grayscale conversion, denoising, thresholding, deskewing, and contrast enhancement.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Continue to OCR", use_container_width=True, type="primary"):
                with st.spinner("Loading OCR model..."):
                    try:
                        # Get OCR engine (cached)
                        engine = get_ocr_engine()
                        
                        log_processing_step(logger, "OCR Text Extraction")
                        # Extract text from image
                        ocr_text = engine.ocr(
                            image=st.session_state.uploaded_image,
                            max_new_tokens=1024
                        )
                        st.session_state.ocr_text = ocr_text
                        
                        log_processing_step(logger, "Layout Analysis")
                        # Analyze document layout
                        layout = analyze_layout(st.session_state.uploaded_image)
                        st.session_state.layout = layout
                        
                        # Move to next step
                        st.session_state.step = 3
                        st.rerun()
                        
                    except Exception as e:
                        log_error(logger, e, "OCR Processing")
                        st.error(f"⚠️ Error processing image: {str(e)}")
    
    # Step 3: OCR Detection Results
    elif current_step == 3:
        st.subheader("Step 3: Text & Layout Analysis Results")
        
        ocr_text = st.session_state.ocr_text
        layout = st.session_state.layout
        
        # Statistics
        text_lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lines Detected", len(layout.lines))
        with col2:
            st.metric("Text Lines", len(text_lines))
        with col3:
            bold_count = sum(1 for line in layout.lines if line.style.bold)
            st.metric("Bold Lines", bold_count)
        with col4:
            italic_count = sum(1 for line in layout.lines if line.style.italic)
            st.metric("Italic Lines", italic_count)
        
        # Show layout debug visualization
        st.markdown("**Layout Analysis**")
        show_debug = st.checkbox("Show detected layout overlay", value=True)
        
        if show_debug:
            debug_img = render_layout_debug(st.session_state.uploaded_image, layout)
            st.image(debug_img, caption="Detected lines with alignment and formatting", use_container_width=True)
        else:
            st.image(st.session_state.uploaded_image, use_container_width=True)
        
        # Detected text
        with st.expander("📝 View Extracted Text"):
            st.text_area(
                "Document Content",
                value=ocr_text,
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("Next: Generate Document", use_container_width=True, type="primary"):
                st.session_state.step = 4
                st.rerun()
    
    # Step 4: Generate Document
    elif current_step == 4:
        st.subheader("Step 4: Generate Document")
        
        ocr_text = st.session_state.ocr_text
        layout = st.session_state.layout
        
        st.info("✨ Ready to generate your formatted Word document")
        
        # Show preview
        st.markdown("**Document Preview**")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Text Content", f"{len(ocr_text)} characters")
            st.metric("Detected Lines", len(layout.lines))
        with col2:
            bold_lines = sum(1 for line in layout.lines if line.style.bold)
            italic_lines = sum(1 for line in layout.lines if line.style.italic)
            st.metric("Bold Lines", bold_lines)
            st.metric("Italic Lines", italic_lines)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("Generate .docx", use_container_width=True, type="primary"):
                with st.spinner("Generating Word document..."):
                    try:
                        log_processing_step(logger, "Document Generation")
                        
                        # Generate document using dual-pipeline approach
                        docx_bytes = build_docx(
                            title="",  # No title by default
                            ocr_text=ocr_text,
                            layout=layout
                        )
                        
                        st.session_state.docx_bytes = docx_bytes
                        st.session_state.step = 5
                        st.rerun()
                    except Exception as e:
                        log_error(logger, e, "Document Generation")
                        st.error(f"Error generating document: {str(e)}")
    
    # Step 5: Download Document
    elif current_step == 5:
        st.subheader("Step 5: Download Document")
        
        st.success("✅ Document Generated Successfully!")
        
        # Download button
        filename = "ocr_output.docx"
        st.download_button(
            label="📥 Download Word Document",
            data=st.session_state.docx_bytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            type="primary"
        )
        
        # Summary statistics
        layout = st.session_state.layout
        ocr_text = st.session_state.ocr_text
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detected Lines", len(layout.lines))
        with col2:
            bold_count = sum(1 for line in layout.lines if line.style.bold)
            st.metric("Bold Lines", bold_count)
        with col3:
            italic_count = sum(1 for line in layout.lines if line.style.italic)
            st.metric("Italic Lines", italic_count)
        
        # Process another image
        st.markdown("---")
        if st.button("🔄 Process Another Image", use_container_width=True):
            reset_session()
            st.rerun()


if __name__ == "__main__":
    main()
