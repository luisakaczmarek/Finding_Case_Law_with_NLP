import streamlit as st
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import plotly.express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import faiss
import concurrent.futures

# Set page config
st.set_page_config(
    page_title="Legal Case Retrieval System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache model loading to avoid reloading on every run
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
    legal_pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")
    summarizer = pipeline("summarization", model=legal_pegasus_model, tokenizer=tokenizer)
    return embedding_model, summarizer

# Cache document loading for faster subsequent runs
@st.cache_data
def load_documents(folder_path):
    """Load documents from individual text files in the specified folder."""
    text_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    documents = {}
    for text_file in tqdm(text_files, desc="Loading documents"):
        # Use the file name (without extension) as the case number
        case_number = os.path.splitext(text_file)[0]
        with open(os.path.join(folder_path, text_file), "r", encoding="utf-8") as file:
            documents[case_number] = file.read()
    return documents

def get_case_url(case_number):
    """Generate URL for a case number."""
    prefix = '-'.join(case_number.split('-')[:2])
    return f"https://law.justia.com/cases/federal/appellate-courts/cit/{prefix}/{case_number}.html"

def build_faiss_index(embeddings):
    """
    Build a FAISS index using normalized embeddings.
    """
    embeddings = np.array(embeddings).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-10)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)  # Use inner product on normalized vectors for cosine similarity
    index.add(normalized_embeddings)
    return index, normalized_embeddings

def find_most_similar_cases(query_text, index, normalized_embeddings, case_numbers, embedding_model, k=5):
    """
    Given the query text, finds the top k similar cases using the FAISS index.
    """
    query_embedding = embedding_model.encode(query_text)
    query_embedding = np.array(query_embedding).astype("float32")
    query_norm = np.linalg.norm(query_embedding)
    if query_norm > 0:
        query_embedding /= query_norm
    query_embedding = np.expand_dims(query_embedding, axis=0)
    scores, indices = index.search(query_embedding, k)
    top_k_scores = scores[0]
    top_k_indices = indices[0]
    top_k_case_numbers = [case_numbers[i] for i in top_k_indices]
    return top_k_case_numbers, top_k_scores

def summarize_case(case_text, summarizer):
    """Summarize a case using the legal pegasus model."""
    try:
        summary = summarizer(case_text, max_length=130, min_length=30, do_sample=False)
        return summary[0]["summary_text"].strip()
    except Exception as e:
        return f"⚠️ Failed to summarize case: {str(e)}"

def visualize_embeddings(embeddings, case_numbers):
    """Visualize embeddings using t-SNE."""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig = px.scatter(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        title="Visualization of Case Embeddings using t-SNE",
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
    )
    
    return fig

def main():
    st.title("⚖️ Legal Case Retrieval and Summarization System")
    st.markdown("""
    This system helps you find and summarize relevant legal cases based on your query.
    It uses semantic search with legal-specific embeddings and provides summaries using a legal-trained LLM.
    """)
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of cases to retrieve", 1, 10, 3)
        show_visualization = st.checkbox("Show embeddings visualization", value=True)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses:
        - [sbert-legal-xlm-roberta-base](https://huggingface.co/Stern5497/sbert-legal-xlm-roberta-base) for semantic search
        - [legal-pegasus](https://huggingface.co/nsi319/legal-pegasus) for summarization
        """)
    
    # Load models and documents
    embedding_model, summarizer = load_models()
    folder_path = os.path.join("data", "extracted_texts")
    documents = load_documents(folder_path)
    st.write(f"Loaded {len(documents)} documents.")
    
    # Load precomputed embeddings from pickle if available
    pickle_filename = "case_embeddings.pkl"
    if os.path.exists(pickle_filename):
        with st.spinner("Loading precomputed case embeddings..."):
            with open(pickle_filename, "rb") as f:
                data = pickle.load(f)
            case_numbers = data["case_numbers"]
            case_embeddings = data["embeddings"]
    else:
        with st.spinner("Computing case embeddings..."):
            case_embeddings = embedding_model.encode(list(documents.values()), show_progress_bar=True)
            case_numbers = list(documents.keys())
    
    # Build a FAISS index for efficient similarity search
    faiss_index, normalized_embeddings = build_faiss_index(case_embeddings)
    
    # User query input
    query = st.text_area("Enter your legal query:", placeholder="E.g., 'Which cases relate to patent infringement?'")
    
    if st.button("Search Cases"):
        if not query.strip():
            st.error("Please enter a query.")
            return
            
        with st.spinner("Finding relevant cases..."):
            similar_cases, similarity_scores = find_most_similar_cases(
                query, faiss_index, normalized_embeddings, case_numbers, embedding_model, k=top_k
            )
        
        st.subheader(f"Top {top_k} Most Relevant Cases")
        
        # Generate summaries in parallel
        summaries = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {executor.submit(summarize_case, documents[cn], summarizer): cn for cn in similar_cases}
            for future in concurrent.futures.as_completed(future_to_case):
                case_num = future_to_case[future]
                try:
                    summaries[case_num] = future.result()
                except Exception as exc:
                    summaries[case_num] = f"⚠️ Failed to summarize: {exc}"
        
        # Display the results
        for i, (case_num, score) in enumerate(zip(similar_cases, similarity_scores), 1):
            with st.expander(f"Case #{i}: {case_num} (Similarity: {score:.2f})"):
                url = get_case_url(case_num)
                st.markdown(f"[View full case on Justia]({url})")
                st.subheader("Summary")
                st.write(summaries.get(case_num, "No summary available."))
                st.subheader("Excerpt")
                excerpt = documents[case_num][:500] + "..." if len(documents[case_num]) > 500 else documents[case_num]
                st.text(excerpt)
        
        # Visualize the embeddings if enabled
        if show_visualization:
            st.subheader("Case Embeddings Visualization")
            fig = visualize_embeddings(case_embeddings, case_numbers)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
