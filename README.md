🧠 NeuroScan AI — Alzheimer’s Detection using MRI

📌 Overview

NeuroScan AI is an end-to-end deep learning system designed to detect Alzheimer’s disease from MRI scans. The project performs both binary classification (CN vs AD) and multi-class classification (CN, MCI, AD) using a ResNet-based architecture.

---

🎯 Problem Statement

To develop a reliable AI-based solution that can analyze MRI scans and classify patients into:

- Cognitively Normal (CN)
- Mild Cognitive Impairment (MCI)
- Alzheimer’s Disease (AD)

---

🚀 Features

- 📥 Upload MRI images
- 🧠 Binary & Multi-class classification
- ⚡ Real-time prediction using Streamlit
- 📊 Confidence score visualization
- 🧹 Data preprocessing & cleaning pipeline
- 📈 Model evaluation with metrics and graphs

---

🏗️ Tech Stack

🔹 Backend (ML/DL)

- PyTorch
- MONAI (Medical Imaging Framework)
- NumPy, Pandas

🔹 Frontend

- Streamlit
- Custom CSS

🔹 Utilities

- OpenCV
- pydicom
- scikit-learn
- matplotlib
- tqdm

---

🧠 Model Architecture

- ResNet18 (CNN-based architecture)
- Residual connections for deep feature learning
- Separate models for:
  - Binary Classification (CN vs AD)
  - Multi-class Classification (CN, MCI, AD)

---

📂 Dataset

- MRI scans in DICOM format
- Converted to 2D grayscale images
- Labeled using metadata (CSV file)

---

⚙️ Data Preprocessing

- DICOM to image conversion
- Slice extraction
- Intensity normalization
- Image resizing (224×224)
- Removal of corrupted/blank images
- Patient-level data splitting (to avoid data leakage)

---

📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

🧪 Results

- Binary Classification Accuracy: ~90%+
- Multi-class Classification Accuracy: ~60–70% (patient-level validation)

---

🌐 Deployment

- Built using Streamlit
- Supports real-time inference
- User-friendly interface

---

▶️ How to Run

1. Clone Repository

git clone https://github.com/your-username/neuroscan-ai.git
cd neuroscan-ai

2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate

3. Install Requirements

pip install -r requirements.txt

4. Run App

streamlit run app.py

---

📁 Project Structure

├── app.py
├── evaluate.py
├── models/
│   ├── binary_resnet_monai.pth
│   └── multiclass_resnet_monai.pth
├── dataset/
├── requirements.txt
└── README.md

---

💡 Novelty

- End-to-end pipeline from raw MRI data to deployment
- Patient-level data splitting to prevent leakage
- Dual-model approach for screening and staging
- Integration of medical imaging framework (MONAI)

---

⚠️ Disclaimer

This project is for research and educational purposes only and should not be used for clinical diagnosis.

---

👨‍💻 Author

CodeCrawlers

---

⭐ Acknowledgements

- MONAI Framework
- PyTorch
- Open-source medical imaging datasets
