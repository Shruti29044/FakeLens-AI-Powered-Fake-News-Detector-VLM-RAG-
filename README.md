🧠 FakeLens: Multimodal Fake News Detector
FakeLens is a Colab-friendly, multimodal misinformation detection system that integrates:

📝 Text Classification using BERT

🔎 Keyword Extraction using TF-IDF

🧠 RAG-based Response Generation (FAISS + Sentence-BERT + FLAN-T5)

🖼️ Image Forgery Detection using a CNN (ResNet18)

🔗 Image-Text Similarity using OpenAI's CLIP

📷 Image Captioning using BLIP

It runs entirely in Google Colab, combining Streamlit UI + Torch/Transformers pipeline, with no backend server required.

🚀 Try it on Google Colab
Click below to launch the notebook in Colab:


📦 Dependencies
The app uses the following packages:

bash
Copy
Edit
streamlit
torch
transformers
sentence-transformers
faiss-cpu
Pillow
scikit-learn
git+https://github.com/openai/CLIP.git
These are installed automatically in the notebook.

🛠️ Features
Component	Model	Description
Text Classification	BERT (IMDb fine-tuned)	Classifies input claim as real or fake
Keyword Extraction	TF-IDF	Highlights key terms from the claim
RAG Response	FAISS + Sentence-BERT + FLAN-T5	Retrieves evidence and generates contextual reasoning
Image Forgery Detection	ResNet18 CNN	Detects image tampering
CLIP Similarity	CLIP (ViT-B/32)	Measures image-text semantic match
Image Captioning	BLIP	Describes the uploaded image contextually

📸 Sample Input
Text:

“The Earth is flat and vaccines are harmful.”

Image:
Example Moon Image

🖥️ How to Use in Colab
Upload Image & Enter Claim

Click "Analyze"

View:

🔑 Keywords

📄 Text classification

🖼️ Forgery detection

🔗 CLIP similarity

📚 Generated response

🧠 Image caption

📁 Folder Structure (if exporting)
bash
Copy
Edit
├── app.py                # Streamlit app
├── models/               # (Optional) Pretrained model caching
├── assets/               # Sample images
🧪 Example Claims to Try
“Vaccines contain microchips for tracking.”

“5G towers spread coronavirus.”

“The Moon landing was staged in Hollywood.”

⚠️ Disclaimer
This tool is for research and educational purposes only. It does not provide absolute truth and should not be used as a final decision system.

