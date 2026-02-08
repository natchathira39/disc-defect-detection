# 🔧 Disc Defect Detection System

AI-powered web application for detecting metal disc surface defects using Deep Learning.

## 📋 Overview

Multi-class classification model that detects 5 types of defects:
- ✅ **Good** - No defects
- ⚠️ **Patches** - Surface patches
- ❌ **Rolled Pits** - Small holes/depressions  
- ⚠️ **Scratches** - Linear defects
- ❌ **Waist Folding** - Material warping

## 🚀 Deployment Steps

### 1. Upload Model to Google Drive
- Upload your `best_model_XXXXXXXX.h5` file to Google Drive
- Right-click → Share → "Anyone with the link can view"
- Copy the file ID from share link:
  ```
  https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
  ```

### 2. Create GitHub Repository
- Create new repository (e.g., `Disc-Defect-Detection`)
- Upload these files:
  - `app.py`
  - `requirements.txt`
  - `README.md`
- Make repository **Public**

### 3. Update File ID in app.py
- Open `app.py`
- Line 14: Replace `YOUR_FILE_ID_HERE` with your actual file ID
  ```python
  GOOGLE_DRIVE_FILE_ID = "15NeEfT7106PH6RnolnhPdHWwHLMz49yC"
  ```

### 4. Deploy to Streamlit Cloud
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Click "New app"
- Select your repository
- Main file: `app.py`
- Click "Deploy"

## 🛠️ Local Testing

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Model Info

- **Architecture:** MobileNetV2 Transfer Learning
- **Input Size:** 224×224 RGB
- **Classes:** 5
- **Framework:** TensorFlow/Keras

## 📁 Project Structure

```
Disc-Defect-Detection/
├── app.py              # Streamlit app
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🔧 Troubleshooting

**Model won't download?**
- Check Google Drive link is "Anyone with the link can view"
- Verify file ID is correct

**Wrong predictions?**
- Ensure image is clear and well-lit
- Check defect type is in training classes

## 👨‍💻 Author

Your Name - [GitHub](https://github.com/YOUR_USERNAME)
