@echo off
echo ================================================
echo OCR with Formatting Preservation - Phase 1
echo ================================================
echo.
echo Installing/Updating dependencies...
pip install -r requirements.txt
echo.
echo Starting Streamlit application...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ================================================
echo.
streamlit run src/streamlit_app.py
