#!/usr/bin/env python3
"""
Build script for creating Windows executable from Streamlit app
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_executable():
    """Build the executable using PyInstaller"""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    app_dir = current_dir / "app"
    
    # Create the PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Create a single executable file
        "--windowed",  # Don't show console window on Windows
        "--name=BankReconciliationAI",  # Name of the executable
        "--add-data", f"{app_dir}:app",  # Include the app directory
        "--additional-hooks-dir=.",  # Use our custom hook file
        "--hidden-import=streamlit",
        "--hidden-import=pandas",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.ensemble",
        "--hidden-import=sklearn.ensemble._forest",
        "--hidden-import=sklearn.tree",
        "--hidden-import=sklearn.tree._utils",
        "--hidden-import=sklearn.utils",
        "--hidden-import=sklearn.utils._cython_blas",
        "--hidden-import=sklearn.neighbors.typedefs",
        "--hidden-import=sklearn.neighbors.quad_tree",
        "--hidden-import=sklearn.tree._utils",
        "--hidden-import=sklearn.utils._typedefs",
        "--hidden-import=numpy",
        "--hidden-import=xgboost",
        "--hidden-import=joblib",
        "--hidden-import=openpyxl",
        "--hidden-import=rapidfuzz",
        "--hidden-import=psycopg2",
        "--hidden-import=cx_Oracle",
        "--collect-all=streamlit",
        "--collect-all=pandas",
        "--collect-all=sklearn",
        "launcher.py"  # Main entry point
    ]
    
    print("Building executable...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run PyInstaller
    result = subprocess.run(cmd, cwd=current_dir, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Build successful!")
        print(f"Executable created in: {current_dir}/dist/BankReconciliationAI.exe")
    else:
        print("❌ Build failed!")
        print("Error output:")
        print(result.stderr)
        return False
    
    return True

if __name__ == "__main__":
    build_executable() 