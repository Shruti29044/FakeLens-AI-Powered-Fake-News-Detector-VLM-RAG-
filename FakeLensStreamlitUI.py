!pip install -q streamlit pyngrok torch torchvision faiss-cpu \
    transformers sentence-transformers Pillow scikit-learn \
    git+https://github.com/openai/CLIP.git
from pathlib import Path

app_code = '''
import torch, clip, faiss, requests
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    pipeline, BlipProcessor, BlipForConditionalGeneration
)

@st.cache_resource
def load_models():
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb").eval()
    retriever = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model, clip_preprocess = clip.load("ViT-B/32")

    cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    cnn.fc = torch.nn.Linear(cnn.fc.in_features, 2)
    cnn.eval()

    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    return bert_tokenizer, bert_model, retriever, clip_model, clip_preprocess, cnn, qa_pipeline, blip_processor, blip_model

bert_tokenizer, bert_model, retriever, clip_model, clip_preprocess, cnn_model, qa_pipeline, blip_processor, blip_model = load_models()

def extract_keywords(texts):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    matrix = vectorizer.fit_transform(texts).todense()
    names = vectorizer.get_feature_names_out()
    return [names[i] for i in matrix[0].argsort().tolist()[0][-5:]]

def classify_text(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "Fake" if pred else "Real"

def detect_forgery(img):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = torch.argmax(cnn_model(tensor), 1).item()
    return "Fake" if pred else "Real"

def compute_clip_similarity(img, text):
    tensor = clip_preprocess(img).unsqueeze(0)
    tokens = clip.tokenize([text])
    with torch.no_grad():
        img_feat = clip_model.encode_image(tensor)
        txt_feat = clip_model.encode_text(tokens)
        sim = torch.nn.functional.cosine_similarity(img_feat, txt_feat).item()
    return round(sim, 3)

def retrieve_and_generate(query):
    docs = ["NASA landed on the moon.", "The Earth is flat.", "Vaccines are safe and effective."]
    doc_embeds = retriever.encode(docs)
    index = faiss.IndexFlatL2(doc_embeds.shape[1])
    index.add(np.array(doc_embeds))
    query_embed = retriever.encode([query])
    _, I = index.search(query_embed, k=2)
    context = " ".join([docs[i] for i in I[0]])
    prompt = f"Claim: {query} Context: {context}"
    result = qa_pipeline(prompt, max_length=64)[0]["generated_text"]
    return result

def generate_blip_caption(img):
    inputs = blip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

st.title("🧠 FakeLens: Multimodal Fake News Detector")

text_input = st.text_area("Enter a suspicious claim:", "The Earth is flat and vaccines are harmful.")
image_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("Analyze"):
    if not image_input:
        st.warning("Please upload an image.")
    else:
        image = Image.open(image_input).convert("RGB")
        with st.spinner("Analyzing..."):
            keywords = extract_keywords([text_input])
            text_class = classify_text(text_input)
            forgery = detect_forgery(image)
            clip_sim = compute_clip_similarity(image, text_input)
            rag = retrieve_and_generate(text_input)
            blip = generate_blip_caption(image)

        st.subheader("📝 Results")
        st.image(image, caption="Uploaded Image", width=300)
        st.write("🔑 Keywords:", ", ".join(keywords))
        st.write("📄 BERT Classification:", text_class)
        st.write("🖼️ CNN Forgery Detection:", forgery)
        st.write("🔗 CLIP Similarity Score:", clip_sim)
        st.write("📚 RAG Output:", rag)
        st.write("🧠 BLIP Caption:", blip)
'''

Path("app.py").write_text(app_code)
print("✅ app.py saved")

from pyngrok import ngrok
import nest_asyncio
import threading
import time

nest_asyncio.apply()

def run():
    !streamlit run app.py --server.headless true --server.enableCORS false &

threading.Thread(target=run).start()
time.sleep(5)  # wait for streamlit to launch

public_url = ngrok.connect(8501)
print("🔗 Your app is live at:", public_url)
