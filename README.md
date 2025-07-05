# research-paper-explainer
# 🧠 Research Paper to Blog-Style Explainer Generator
This project simplifies complex academic research papers—particularly from sources like arXiv or academic conferences—into easy-to-understand blog-style summaries. Using state-of-the-art AI and Natural Language Processing (NLP), the system extracts core components like problem statement, methodology, and key takeaways, and presents them in a way that non-experts can grasp.

### 📍 Summer Training Bootcamp Project

🎓 __Institution:__ SRMCEM, Lucknow

💡 __Internship Theme:__ Artificial Intelligence

👨‍💻 __Team Members:__

- Priyanshi Nishad

- Shruti Verma

- Raghav Pratap Soni

__Open Project on Colab__
🔗https://colab.research.google.com/drive/1IHpc1kebESZSUxnllRWATPXvldwL5avs?usp=sharing

## 📘 Project Topic
### "AI-Powered Academic Research Simplifier"

- __Input:__ PDF research papers from arXiv, conferences, or academic journals
- __Output:__ A blog-style summary that covers the research problem, approach/methodology, and key results, tailored for readers without a technical background.

## 🎯 Goal:
Bridge the gap between technical researchers and the public by making academic insights more accessible and digestible.

## 🧰 Libraries and Tools Used
- __PyPDF2__ – Extracts clean text from academic PDF files.

- __InstructorEmbedding__ – Hugging Face model (hkunlp/instructor-xl) that generates context-aware embeddings with task-specific instructions.

- __faiss-cuda__ – Enables fast semantic similarity search among text chunks.

- __transformers__ – Accesses models like LLaMA for summarization.

- __meta-llama/llama-4-scout-17b-16e-instruct__ -A powerful open-source large language model from Meta, used here to generate conversational, easy-to-read summaries from complex academic text.

- __Streamlit__ –  Powers the interactive web app, allowing users to upload PDFs and instantly view blog-style summaries.

## 🧭 Project Workflow
- __Upload Research Paper__ – User uploads a PDF via the Streamlit interface.

- __Text Extraction__ – PyPDF2 retrieves clean textual content.

- __Text Chunking & Embedding__ – Text is split and encoded using InstructorEmbedding with instructions like "Represent the methodology from this paper".

- __Semantic Search__ – FAISS locates the most relevant chunks for each explainer section

- __Summarization__ – LLaMA 3 Instruct generates concise and readable explanations

- __Display Summary__ – The final output is shown within the Streamlit app in a blog-style format

## ✅ How to Use
1. Open the Colab notebook.

2. Upload any academic research paper in PDF format.

3.Let the system process and summarize key content.

4. Get a blog-style explainer, perfect for sharing or studying.

## 📌 Input & Output Example
### Sample PDF Input:
__Title:__ "Transformer-based Models for Document Summarization"

- __Problem:__ Summarizing long documents is still challenging for NLP models.

- __Method:__ Introduces a hierarchical attention mechanism for long text.

- __Result:__ Achieves state-of-the-art performance on benchmark datasets.

### Blog-Style Output:
- __Problem:__ Long documents are difficult for traditional models to summarize accurately.
- __Approach:__ This paper introduces a layered attention method to better handle complex content.
- __Key Takeaways:__ The proposed model outperforms previous methods on multiple summarization benchmarks.

## 🚀 Future Enhancements
- Support for scanned or image-based papers using OCR.

- Summaries in regional languages like Hindi or Bengali.

- Option to download or share summaries as PDF or blog posts.

- Integration with a Q&A chatbot for interactive learning.

## 📑 License
Developed for academic and learning purposes as part of SRMCEM’s Summer AI Bootcamp 2025. All methods and models are for educational demonstration only.

## 👥 Contributors
- Priyanshi Nishad - Helped in project code on colab

- Shruti Verma - Handled presentation 

- Raghav Pratap Soni - Handled UI (Streamlit)
