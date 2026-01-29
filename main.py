"""
Main Application Entry Point
OCR with Formatting Preservation - Phase 1
Now with Streamlit GUI and comprehensive logging
"""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)


def main():
    """
    Main function to launch the OCR application.
    """
    print("=" * 60)
    print("OCR with Formatting Preservation - Phase 1")
    print("=" * 60)
    print()
    print("Starting Streamlit application...")
    print("The app will open in your default web browser.")
    print()
    print("To run manually, use:")
    print("  streamlit run src/streamlit_app.py")
    print("=" * 60)
    
    # Launch Streamlit app
    import subprocess
    streamlit_path = os.path.join(src_path, 'streamlit_app.py')
    subprocess.run(['streamlit', 'run', streamlit_path])


if __name__ == "__main__":
    main()
