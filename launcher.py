#!/usr/bin/env python3
"""
Launcher script for the Streamlit app executable
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def start_streamlit():
    """Start the Streamlit application"""
    
    # Get the path to the ui.py file
    ui_path = get_resource_path("app/ui.py")
    
    # Set environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Start Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", ui_path,
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    print("Starting Bank Reconciliation AI...")
    print("The application will open in your default web browser.")
    
    # Start Streamlit in a subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for the server to start
    time.sleep(3)
    
    # Open the browser
    try:
        webbrowser.open("http://localhost:8501")
        print("✅ Application is running at http://localhost:8501")
        print("Press Ctrl+C to stop the application")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please open your browser and go to: http://localhost:8501")
    
    # Keep the process running
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping application...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    start_streamlit() 