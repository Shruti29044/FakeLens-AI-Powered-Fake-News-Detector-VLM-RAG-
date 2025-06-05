# ✅ Cell 1: Install dependencies (Colab-friendly and memory optimized)
!pip install -q torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
!pip install -q transformers==4.41.1 sentence-transformers==2.6.1 faiss-cpu==1.7.4 Pillow scikit-learn
!pip install -q git+https://github.com/openai/CLIP.git

# ✅ Cell 2: Import libraries
import gc
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from PIL import Image
import io
import clip
import faiss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import psutil
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# ✅ Cell 3: Track memory usage
print("RAM usage (MB):", psutil.virtual_memory().used / 1024**2)

# ✅ Cell 4: Load models (optimized and no redundant loads)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
bert_model.eval()

retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model, clip_preprocess = clip.load("ViT-B/32")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

class ForgeryModel(nn.Module):
    def __init__(self):
        super(ForgeryModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

cnn_model = ForgeryModel()
cnn_model.eval()

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# ✅ Cell 5: Utility functions

def extract_keywords(texts):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    keywords = [feature_names[i] for i in dense[0].argsort().tolist()[0][-5:]]
    return keywords

def classify_text(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1).item()
    return "Fake" if predicted else "Real"

def detect_forgery(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        predicted = torch.argmax(outputs, 1).item()
    return "Fake" if predicted else "Real"

def compute_clip_similarity(img, text):
    img_tensor = clip_preprocess(img).unsqueeze(0)
    text_tokens = clip.tokenize([text])
    with torch.no_grad():
        image_features = clip_model.encode_image(img_tensor)
        text_features = clip_model.encode_text(text_tokens)
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
    return round(similarity, 3)

def retrieve_and_generate(query):
    docs = ["NASA landed on the moon.", "The Earth is flat.", "COVID-19 vaccines are effective."]
    doc_embeds = retriever_model.encode(docs)
    index = faiss.IndexFlatL2(doc_embeds.shape[1])
    index.add(np.array(doc_embeds))
    query_embed = retriever_model.encode([query])
    D, I = index.search(query_embed, k=2)
    context = " ".join([docs[i] for i in I[0]])
    prompt = f"Claim: {query} Context: {context}"
    response = qa_pipeline(prompt, max_length=64)[0]['generated_text']
    return response

def generate_blip_caption(img):
    inputs = blip_processor(images=img, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# ✅ Cell 6: Cleanup to free memory (if needed)
def clean_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    torch.cuda.empty_cache()
    print("🔁 Memory cleaned.")

# 🧪 Sample fake news claim and associated image
sample_text = "The Earth is flat and vaccines are harmful."
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/640px-FullMoon2010.jpg"

# 🖼️ Load image from URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# 🚀 Run inference modules
keywords = extract_keywords([sample_text])
text_result = classify_text(sample_text)
image_result = detect_forgery(image)
clip_score = compute_clip_similarity(image, sample_text)
rag_output = retrieve_and_generate(sample_text)
blip_caption = generate_blip_caption(image)

# 📸️ Display results
print("🔑 Keywords:", keywords)
print("📄 Text Classification (BERT):", text_result)
print("🖼️ Image Forgery Detection (CNN):", image_result)
print("📷📝 CLIP Similarity Score:", clip_score)
print("📚 RAG Output:", rag_output)
print("🌟 BLIP Caption:", blip_caption)
