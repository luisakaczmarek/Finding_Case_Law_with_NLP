import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import numpy as np
import os

# -------------------------
# Load Models
# -------------------------
embedding_model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base")

# Summarizer (Legal Pegasus)
sum_tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")
summarizer = pipeline("summarization", model=sum_model, tokenizer=sum_tokenizer)

# Optimized LLM for generation (OpenChat - faster and ungated)
llm_model_id = "openchat/openchat-3.5-1210"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id, device_map="auto")
llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)

# -------------------------
# Load Documents
# -------------------------
@st.cache_data
def load_documents(folder_path="./data/extracted_texts"):
    documents = {}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                case_number = os.path.splitext(file)[0]
                content = f.read().strip()
                if content:
                    documents[case_number] = content
    return documents

documents = load_documents()
case_numbers = list(documents.keys())
case_texts = list(documents.values())
case_embeddings = embedding_model.encode(case_texts, show_progress_bar=True)

# -------------------------
# Helper Functions
# -------------------------
def truncate_text(text, max_chars=2000):
    return text[:max_chars] + "..." if len(text) > max_chars else text

def find_most_similar_cases(query_text, k=3):
    query_embedding = embedding_model.encode(query_text)
    similarities = util.cos_sim(query_embedding, case_embeddings)[0]
    top_k_indices = np.argsort(similarities.numpy())[::-1][:k]
    return [(case_numbers[i], case_texts[i]) for i in top_k_indices]

def summarize_cases(cases):
    results = []
    for case_number, case_text in cases:
        summary = summarizer(case_text, max_length=130, min_length=30, do_sample=False)
        results.append((case_number, summary[0]['summary_text']))
    return results

def generate_answer(query, cases):
    context = "\n\n".join([
        f"### Case {i+1} ({cn}):\n{truncate_text(ct)}"
        for i, (cn, ct) in enumerate(cases)
    ])
    prompt = f"""You are a legal assistant. Answer the user's legal question based only on the context below.

If the answer is not in the context, reply: \"The answer is not available in the provided documents.\"

---

### User Question:
{query}

---

### Context:
{context}

---

### Your Answer:
"""
    response = llm_pipeline(prompt, max_new_tokens=200, do_sample=False, temperature=0.0)
    return response[0]['generated_text'].replace(prompt, '').strip()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Legal Case Assistant", layout="wide")
st.title("⚖️ Legal Case Assistant")

query = st.text_input("🔍 Enter your legal question:", placeholder="e.g. Can a tenant be evicted without notice?")
top_k = st.slider("Number of similar cases to retrieve:", min_value=1, max_value=10, value=3)

if st.button("Search & Summarize") and query.strip():
    with st.spinner("Retrieving relevant cases and generating response..."):
        top_cases = find_most_similar_cases(query, k=top_k)
        summaries = summarize_cases(top_cases)
        answer = generate_answer(query, top_cases)

    st.subheader("📚 Top Case Summaries")
    for i, (case_number, summary_text) in enumerate(summaries, start=1):
        st.markdown(f"**🔹 Case {i}: {case_number}**")
        st.markdown(f"> {summary_text}")
        st.markdown("---")

    st.subheader("🧠 AI Answer Based on Full Case Context")
    st.markdown(f"**Question:** {query}")
    st.markdown(f"**Answer:** {answer}")
