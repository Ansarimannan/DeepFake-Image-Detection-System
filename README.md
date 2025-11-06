# Deepfake Image Detection

Binary classifier to detect real vs fake (deepfake) faces using MobileNetV2 transfer learning.

## Quick Start

### 1. Setup (First Time Only)

**Option A: Use Launcher (Recommended)**
```
Double-click: launcher.bat
Choose option 1: Setup
```

**Option B: Direct**
```
Double-click: setup.bat
```

This will:
- Check for Python 3.11
- Create virtual environment
- Install all packages (TensorFlow, etc.)

### 2. Run Notebook

**Interactive (Recommended for first time):**
```
Double-click: open.bat
```
Opens notebook in browser for step-by-step execution.

**Automated:**
```
Double-click: run.bat
```
Executes entire notebook automatically.

## Requirements

- **Python 3.11** (required for TensorFlow)
  - Download: https://www.python.org/downloads/release/python-3119/
  - Check "Add Python 3.11 to PATH" during installation

## Files

**Main Entry Point:**
- `launcher.bat` - Interactive menu with all options (recommended)

**Individual Scripts:**
- `setup.bat` - Initial setup (creates venv, installs packages)
- `open.bat` - Open notebook interactively
- `run.bat` - Run notebook automatically
- `cleanup.bat` - Remove locked virtual environment
- `check_tensorflow.bat` - Verify TensorFlow installation
- `setup_kaggle.bat` - Setup Kaggle API for auto-download
- `download_dataset.bat` - Download dataset from Kaggle

## Dataset Setup

### Option 1: Kaggle API (Recommended - Automatic)

1. **Setup Kaggle API:**
   - Run `launcher.bat` → option 7 (Setup Kaggle API)
   - Or double-click `setup_kaggle.bat`
   - This opens Kaggle settings and guides you through downloading `kaggle.json`

2. **Place kaggle.json:**
   - Location: `C:\Users\YourUsername\.kaggle\kaggle.json`
   - Create `.kaggle` folder if it doesn't exist

3. **Download dataset:**
   - Run `launcher.bat` → option 3 (Download Dataset)
   - Or the notebook will download automatically when you run the data acquisition cell

### Option 2: Manual ZIP File

1. **Download the dataset:**
   - Go to: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection
   - Click "Download" (requires Kaggle account)
   - Save the ZIP file anywhere

2. **Load in notebook:**
   - Find the "HELPER: Load dataset from ZIP file" cell
   - Uncomment and modify the path:
     ```python
     load_from_zip(r'E:\Downloads\real-and-fake-face-detection.zip')
     ```
   - Run that cell, then re-run the data acquisition cell

### Option 3: Use Your Own Dataset

If you have your own dataset:
- Must have `real/` and `fake/` folders
- Each folder should contain images (.jpg, .png)
- Use `load_from_zip()` function with your ZIP file path

### Dataset Structure

The dataset should look like this:
```
data/
  real_fake/
    real/
      image1.jpg
      image2.jpg
      ...
    fake/
      image1.jpg
      image2.jpg
      ...
```

## Troubleshooting

**TensorFlow not found:**
- Run `launcher.bat` → option 1 (Setup) to install everything
- Or check installation: `launcher.bat` → option 6 (Check TensorFlow)

**Python version error:**
- Must use Python 3.11 (not 3.12+)
- Run `launcher.bat` → option 1 (Setup) to create venv with correct version

**Notebook won't run:**
- Make sure virtual environment is activated
- Check TensorFlow: `launcher.bat` → option 6 (Check TensorFlow)

**"Dataset not found" error:**
- Make sure you've either:
  1. Set up Kaggle API and re-run the cell
  2. Used `load_from_zip()` with correct path
  3. Manually extracted dataset to `./data/real_fake/`

**Kaggle download fails:**
- Check `kaggle.json` is in correct location (`%USERPROFILE%\.kaggle\`)
- Verify Kaggle account has access to the dataset
- Try manual ZIP download instead

**ZIP file not found:**
- Use raw string: `r'C:\path\to\file.zip'`
- Or forward slashes: `'C:/path/to/file.zip'`
- Check file path is correct

## Output

After training, you'll get:
- `best_mnv2.h5` - Trained model
- `label_map.json` - Class labels
- `Deepfake Image Detection — Simple Transfer Learning_executed.ipynb` - Results

