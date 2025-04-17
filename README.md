# Legal Case Retrieval and Summarization System

This repository contains a system for retrieving and summarizing legal cases using advanced NLP models. The system combines semantic search capabilities with abstractive summarization to help users find and understand relevant legal precedents.

---

## 🧠 Overview

The application implements two independent NLP approaches:

- **Case Retrieval using legal-xlm-RoBERTa**  
  A multilingual variant of RoBERTa fine-tuned on legal domain data. The model encodes legal documents and user queries into dense vector representations, enabling semantic search beyond simple keyword matching.

- **Abstractive Summarization using Legal-Pegasus**  
  A domain-adapted version of Pegasus designed specifically for summarizing legal texts. Legal-Pegasus generates concise, understandable summaries of lengthy legal documents.

Together, these models create a powerful tool that helps users discover relevant legal cases and quickly understand their content.

---

## 🗂️ Repository Structure
```bash
.
├── notebooks/
│   ├── 0_web_scraping_cases.ipynb   # Web scraping legal cases
│   ├── 1_convert_cases_pdf_to_txt.ipynb  # Converting PDFs to text
│   └── 2_main.ipynb                 # Core implementation and analysis
├── README.md 
├── app.py # Main Streamlit application
├── case_embeddings.pkl                      
├── case_embeddings_to_pickle.ipynb  # Script to create and save embeddings
├── requirements.txt            # Python dependencies
└── extracted_texts.zip
```
---

## 🛠️ Installation Instructions

### 📁 Clone the repository

```bash
git clone https://github.com/yourusername/legal-case-retrieval.git
cd legal-case-retrieval
```

### 🧪 Create a virtual environment and activate it

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```
### 📦 Install the required dependencies

```bash
pip install -r requirements.txt
```
---

## 🚀 Using the Application 
### ▶️ Start the Streamlit application

```bash
streamlit run app.py
```

- Open your browser and navigate to the URL displayed in the terminal (typically `http://localhost:8501`)
- Enter your legal query in the search box and hit **"Search Cases"**
- Review the retrieved cases and their summaries

---

### ✨ Key Features

- **Semantic Search**: Find cases based on meaning rather than just keywords  
- **Abstractive Summarization**: Get concise summaries of lengthy legal documents  
- **Interactive Interface**: Explore cases with an intuitive web interface  
- **Source Attribution**: All information is linked to source documents  

---
## 🔍 Retrieval Model: `legal-xlm-RoBERTa`

- **Architecture**: Transformer-based encoder (RoBERTa variant)  
- **Training**: Fine-tuned on multilingual legal corpora  
- **Function**: Converts text into dense vector embeddings for semantic similarity search  
- **Advantage**: Can understand semantic relationships between legal concepts even when different terminology is used  

## ✂️ Summarization Model: `Legal-Pegasus`

- **Architecture**: Encoder-decoder transformer with attention mechanisms  
- **Training**: Adapted from Pegasus with additional fine-tuning on legal documents  
- **Function**: Generates abstractive summaries that capture key information from lengthy documents  
- **Advantage**: Produces readable summaries that maintain legal accuracy and relevance  

---

## 📥 Data Collection

The system uses legal cases from the **U.S. Court of International Trade (CIT)** between 2023 and 2025. Data was collected using the following process:

- 🔎 **Web scraping** from [law.justia.com](https://law.justia.com) using Selenium  
  *(See `0_web_scraping_cases.ipynb`)*

- 📄 **Converting PDF documents to text**  
  *(See `1_convert_cases_pdf_to_txt.ipynb`)*

- 🧠 **Processing and embedding texts for efficient retrieval**  
  *(See `case_embeddings_to_pickle.ipynb`)*
