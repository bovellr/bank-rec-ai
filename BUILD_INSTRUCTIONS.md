# Building Windows Executable

This guide explains how to package the Bank Reconciliation AI application into a standalone Windows executable.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Virtual environment** activated
3. **All dependencies** installed from `requirements.txt`

## Step 1: Install Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install all dependencies including PyInstaller
pip install -r requirements.txt
```

## Step 2: Build the Executable

### Option A: Using the Build Script (Recommended)

```bash
python build_exe.py
```

### Option B: Manual PyInstaller Command

```bash
pyinstaller --onefile --windowed --name=BankReconciliationAI --add-data "app:app" --hidden-import=streamlit --hidden-import=pandas --hidden-import=sklearn --hidden-import=numpy --hidden-import=xgboost --hidden-import=joblib --hidden-import=openpyxl --hidden-import=rapidfuzz --hidden-import=psycopg2 --hidden-import=cx_Oracle --collect-all=streamlit --collect-all=pandas --collect-all=sklearn launcher.py
```

## Step 3: Locate the Executable

After successful build, you'll find the executable at:
```
dist/BankReconciliationAI.exe
```

## Step 4: Distribution

1. **Copy the executable** to any Windows machine
2. **Run the executable** - it will:
   - Start a local web server
   - Automatically open your default browser
   - Display the Bank Reconciliation AI interface

## Troubleshooting

### Common Issues:

1. **"Missing module" errors**: Ensure all dependencies are in `requirements.txt`
2. **"sklearn.ensemble could not be resolved"**: This is fixed with the custom hook file (`hook-sklearn.py`)
3. **Large file size**: This is normal for PyInstaller - it bundles Python runtime
4. **Antivirus warnings**: Some antivirus software may flag PyInstaller executables
5. **Port conflicts**: If port 8501 is busy, the app will show an error

### File Size Optimization:

The executable will be large (100-200MB) because it includes:
- Python runtime
- All dependencies (pandas, sklearn, streamlit, etc.)
- Required libraries

### Alternative Approach for Smaller Size:

If file size is a concern, consider:
1. **Using Docker** for distribution
2. **Creating a web installer** that downloads dependencies
3. **Using cx_Freeze** instead of PyInstaller (may be smaller)

## Testing the Executable

1. **Test on the build machine** first
2. **Test on a clean Windows machine** without Python installed
3. **Verify all features work** (file upload, reconciliation, etc.)

## Notes

- The executable creates a temporary directory when run
- It requires internet access for some Streamlit components
- The app runs on `http://localhost:8501`
- Users can close the browser but the server continues running
- Use Ctrl+C in the console to stop the application 