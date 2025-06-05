# FakeLens: Multimodal Fake News Detector

FakeLens is a Google Colab-ready app that detects misinformation using both text and images. It combines models like BERT, CLIP, BLIP, CNN, and a RAG-like system for contextual verification — all accessible through a Streamlit interface.

## 🔧 What it does

* Classifies claims as Real or Fake using a BERT classifier.
* Extracts important keywords using TF-IDF.
* Checks image and text similarity using CLIP.
* Detects forged images with a CNN-based image classifier (ResNet18).
* Generates contextual responses using a Retrieval-Augmented Generator (Sentence-BERT + FLAN-T5).
* Generates image captions using BLIP.

## 📦 Dependencies

The code installs all required packages automatically, including:

* torch
* torchvision
* transformers
* sentence-transformers
* faiss-cpu
* Pillow
* scikit-learn
* streamlit
* git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)

## 🚀 How to run in Google Colab

1. Open the notebook in Colab.
2. Run all cells to install dependencies and load models.
3. When prompted, a public URL will be created using localtunnel.
4. Open that URL in a new tab to use the web app.

## 💡 Sample claims to try

* The Earth is flat and vaccines are harmful.
* COVID-19 was caused by 5G radiation.
* NASA never landed on the moon.
* Drinking bleach cures coronavirus.
* Climate change is a hoax.

## ⚠️ Note

This tool is for research and education only. It doesn’t guarantee accuracy and should not be used for high-stakes decision-making.
