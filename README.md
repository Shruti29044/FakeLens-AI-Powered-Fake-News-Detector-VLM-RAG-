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

⚠️ Challenges Faced
1. Model Size and Memory Management
Loading multiple heavy models (BERT, CLIP, BLIP, ResNet, Sentence-BERT) in a Colab session risked crashing due to limited RAM (typically 12–16GB).

Had to clear unused variables, use @torch.no_grad(), and offload models from GPU to CPU where possible.

2. CLIP Integration Issues
The CLIP model from OpenAI isn't a standard transformers model.

Required git+https://github.com/openai/CLIP.git and careful management of clip.tokenize and clip_preprocess.

3. Streamlit in Colab
Streamlit isn’t natively supported in Colab.

Required pyngrok and nest_asyncio to manually launch and tunnel a Streamlit server.

Faced errors like missing ScriptRunContext or Streamlit warnings when not running via streamlit run.

4. Ngrok / LocalTunnel Auth Errors
ngrok and localtunnel sometimes required manual auth tokens or IP passwords.

Misconfigured tunnel options (port, authtoken) led to ERR_NGROK_4018 and YAML unmarshalling errors.

5. FastAPI Conflicts
Attempting to run both FastAPI and Streamlit in the same Colab script led to ModuleNotFoundError: No module named 'app'.

Realized FastAPI is not practical inside Colab due to ASGI conflicts and the uvicorn lifecycle.

6. Model Loading Delays
First-time model loads (e.g., BLIP, FLAN-T5, ResNet18) took several minutes and occasionally failed if the runtime disconnected or quota exceeded.

7. Token Mismatch Errors
HuggingFace token warnings appeared when accessing gated models or rate-limited APIs without authentication.

8. Integration Complexity
Combining five different AI models—each with different inputs, outputs, preprocessing steps—required careful orchestration and error handling.

Especially challenging to align vision-language outputs across CLIP, BLIP, and text-based RAG.

