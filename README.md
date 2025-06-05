# FakeLens-AI-Powered-Fake-News-Detector-VLM-RAG-



FakeLens is a comprehensive AI-driven system designed to detect and classify misinformation across text and images by leveraging the latest advances in multimodal learning. This project integrates Vision-Language Models (VLMs) with Retrieval-Augmented Generation (RAG) to analyze, interpret, and evaluate the authenticity of multimedia content found in news articles and social media posts.

Multimodal Architecture:
At the core of FakeLens lies a robust multimodal architecture that combines both visual and textual data processing. For visual content, the system utilizes pre-trained CLIP and BLIP models to extract semantic embeddings and cross-modal features, enabling image-text alignment and relevance evaluation. For textual content, the system incorporates a RAG pipeline combining GPT-based language modeling with vector-based retrieval using FAISS and ChromaDB. This hybrid design allows the system to compare questionable content against a curated knowledge base of verified information, enhancing factual consistency checks.

Key Technologies and Methods:

Text Analysis:

Employed BERT for contextual text classification, improving the model’s ability to detect subtle linguistic cues associated with fake news.

Used TF-IDF (Term Frequency–Inverse Document Frequency) to extract salient keywords for topic modeling and relevance scoring.

Integrated RAG to retrieve supporting or contradicting evidence from external sources, enabling deeper cross-verification.

Image Analysis:

Implemented CNN-based image forgery detection to identify tampered or manipulated visuals, using spatial frequency patterns and anomaly detection techniques.

Cross-validated image content with textual narratives using CLIP/BLIP to identify inconsistencies and mismatches.

Performance & Evaluation:
Through extensive testing on benchmark datasets and curated misinformation corpora, FakeLens achieved an overall accuracy of 87% in detecting fake news, significantly outperforming unimodal baselines. The inclusion of CNN-based image forgery detection and advanced textual analysis resulted in a 20% improvement in classification performance, particularly in detecting coordinated misinformation that spans both image and text domains.

Deployment:
The entire system was deployed using FastAPI for scalable API development and Streamlit for an interactive frontend interface. This allows real-time analysis of user-submitted content and provides visual explanations of detection results, enhancing transparency and user trust.

Impact:
FakeLens serves as a powerful tool in the fight against digital misinformation. Its multimodal capabilities make it suitable for journalistic verification, fact-checking platforms, social media moderation, and academic research on information integrity.
