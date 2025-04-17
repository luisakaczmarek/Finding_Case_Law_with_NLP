import streamlit as st
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer, util  # for cosine similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import plotly.express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import concurrent.futures
import ollama  # Ensure you have the Ollama API available

# Set page config
st.set_page_config(
    page_title="Legal Case Retrieval System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)
import torch

def load_models():
    # Force SentenceTransformer to load on CPU.
    embedding_model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base", device="cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
    # Use low_cpu_mem_usage=True to ensure that weights are loaded properly (rather than as meta tensors)
    legal_pegasus_model = AutoModelForSeq2SeqLM.from_pretrained(
        "nsi319/legal-pegasus",
        low_cpu_mem_usage=True,   # helps avoid meta tensor issues
        torch_dtype=torch.float32
    )
    # After loading, explicitly move the model to CPU.
    legal_pegasus_model = legal_pegasus_model.to("cpu")
    
    # When creating the summarization pipeline, do NOT specify a device parameter.
    summarizer = pipeline("summarization", model=legal_pegasus_model, tokenizer=tokenizer)
    
    st.write("Models loaded.")
    return embedding_model, summarizer

def load_documents(folder_path):
    """Load documents from individual text files in the specified folder."""
    text_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    documents = {}
    for text_file in tqdm(text_files, desc="Loading documents"):
        case_number = os.path.splitext(text_file)[0]
        file_path = os.path.join(folder_path, text_file)
        if not os.path.exists(file_path):
            print(f"Debug: File {file_path} does not exist!")
            continue
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            documents[case_number] = content
            print(f"Debug: Loaded case {case_number} with length {len(content)} and content preview: {content[:50]!r}")
    if not documents:
        print("Debug: No documents were loaded! Check the folder path and file contents.")
    return documents

def get_case_url(case_number):
    """Generate URL for a case number."""
    prefix = '-'.join(case_number.split('-')[:2])
    return f"https://law.justia.com/cases/federal/appellate-courts/cit/{prefix}/{case_number}.html"

def find_most_similar_cases(query_text, embedding_model, case_embeddings, case_numbers, k=5):
    """
    Find the most similar cases based on the cosine similarity of embeddings.
    """
    query_embedding = embedding_model.encode(query_text)
    similarities = util.cos_sim(query_embedding, case_embeddings)[0]
    # st.write("Debug: Similarities computed", similarities)
    top_k_indices = np.argsort(similarities.numpy())[::-1][:k]
    top_cases = [case_numbers[i] for i in top_k_indices]
    top_scores = [similarities.numpy()[i] for i in top_k_indices]
    #st.write(f"Debug: Top cases: {top_cases}")
    return top_cases, top_scores

def safe_summarize_first_part_by_tokens(case_text, summarizer, max_tokens=1024):
    """
    Summarizes the first part of a legal case by truncating its tokenized input to
    at most max_tokens tokens. This accounts for special tokens so that the final 
    token count does not exceed max_tokens.
    
    Parameters:
      case_text (str): Full text of the legal case.
      summarizer: A Hugging Face summarization pipeline with an associated tokenizer.
      max_tokens (int): Maximum allowed tokens, including special tokens (default: 1024).
      
    Returns:
      (str): The summary, or an error message if summarization fails.
    """
    # Obtain the tokenizer from the summarizer's pipeline.
    tokenizer = summarizer.tokenizer
    
    # Encode the full text with special tokens.
    tokens = tokenizer.encode(case_text, add_special_tokens=True)
    original_token_count = len(tokens)
    
    # Determine how many special tokens are added by the tokenizer.
    special_tokens_count = tokenizer.num_special_tokens_to_add(pair=False)
    
    # If the total number of tokens is too high, truncate so that total tokens ≤ max_tokens.
    if original_token_count > max_tokens:
        allowed_tokens = max_tokens - special_tokens_count
        tokens = tokens[:allowed_tokens]
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        truncated_text = case_text
    
    # Re-encode truncated text to verify the final token count.
    final_token_count = len(tokenizer.encode(truncated_text, add_special_tokens=True))
    print(f"Final token count for summarization: {final_token_count}")
    
    try:
        summary_output = summarizer(truncated_text, max_length=130, min_length=30, do_sample=False)
        if isinstance(summary_output, list) and summary_output:
            summary_text = summary_output[0].get("summary_text", "").strip()
            if summary_text:
                return summary_text
        return "[No summary generated.]"
    except Exception as e:
        return f"⚠️ Failed to summarize case: {str(e)}"

    """
    Summarizes the first part of a legal case by first truncating its input to a maximum token length.
    
    Parameters:
      case_text (str): Full text of the legal case.
      summarizer: A Hugging Face summarization pipeline (with an associated tokenizer).
      max_tokens (int): Maximum number of tokens allowed (e.g., 1024).
      
    Returns:
      (str): The summary, or an error message if summarization fails.
    """
    # Get the tokenizer from the summarizer's pipeline
    tokenizer = summarizer.tokenizer
    # Encode the text without truncation first, to check token length.
    tokens = tokenizer.encode(case_text, add_special_tokens=True)
    original_token_count = len(tokens)
    if original_token_count > max_tokens:
        print(f"Truncating input: original token count {original_token_count} exceeds max_tokens {max_tokens}.")
        # Truncate tokens to allowed max.
        tokens = tokens[:max_tokens]
        # Decode back to text (skipping special tokens for clarity)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        truncated_text = case_text

    print(f"Final token count for summarization: {len(tokenizer.encode(truncated_text))}")
    
    try:
        summary_output = summarizer(truncated_text, max_length=130, min_length=30, do_sample=False)
        if isinstance(summary_output, list) and summary_output:
            summary_text = summary_output[0].get("summary_text", "").strip()
            if summary_text:
                return summary_text
        return "[No summary generated.]"
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

# Constants for Ollama generation
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """Splits input text into chunks based on word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    print(f"Debug: Split text into {len(chunks)} chunks.")
    return chunks

def generate_response_for_case(case_text, chunk_size=200, overlap=50):
    """
    Processes a legal case by splitting it into chunks and generating a response for each chunk.
    """
    chunks = split_text_into_chunks(case_text, chunk_size=chunk_size, overlap=overlap)
    chunk_responses = []
    
    for idx, chunk in enumerate(chunks):
        instruction_prompt = f"""You are a helpful legal assistant.
Analyze the following part of a legal case and provide a concise summary or insight.
Do not add any information that is not provided.
Chunk {idx+1}:
{chunk}
"""
        print(f"Debug: Generating response for chunk {idx+1} with {len(chunk.split())} words.")
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'system', 'content': instruction_prompt}],
            stream=True,
        )
        
        response = ""
        for message in stream:
            response += message['message']['content']
        chunk_responses.append(response)
        print(f"Processed chunk {idx+1}/{len(chunks)}")
    
    combined_response = "\n\n".join(chunk_responses)
    return combined_response

def main():
    st.title("⚖️ Legal Case Retrieval and Summarization System")
    st.markdown("""
    This system helps you find and summarize relevant legal cases based on your query.
    It uses semantic search with legal-specific embeddings and provides summaries using a legal-trained LLM.
    """)
    
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of cases to retrieve", 1, 10, 3)
        # show_visualization = st.checkbox("Show embeddings visualization", value=True)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses:
        - [sbert-legal-xlm-roberta-base](https://huggingface.co/Stern5497/sbert-legal-xlm-roberta-base) for semantic search
        - [legal-pegasus](https://huggingface.co/nsi319/legal-pegasus) for summarization
        """)
    
    # Load models and documents
    embedding_model, summarizer = load_models()
    folder_path = "./data/extracted_texts"
    documents = load_documents(folder_path)
    st.write(f"Loaded documents.")
    
    # Filter out any empty documents (0 bytes or whitespace only)
    nonempty_documents = {k: v for k, v in documents.items() if v.strip() != ""}
    st.write(f"Filtered: {len(nonempty_documents)} nonempty documents available.")
    
    # Use the nonempty documents for embeddings and search.
    case_numbers = list(nonempty_documents.keys())
    case_texts = list(nonempty_documents.values())
    
    # Load precomputed embeddings from pickle if available
    pickle_filename = "case_embeddings.pkl"
    if os.path.exists(pickle_filename):
        with st.spinner("Loading precomputed case embeddings..."):
            with open(pickle_filename, "rb") as f:
                data = pickle.load(f)
            # Filter embeddings to only include nonempty documents.
            all_case_numbers = data["case_numbers"]
            all_case_embeddings = data["embeddings"]
            filtered_data = [(cn, emb) for cn, emb in zip(all_case_numbers, all_case_embeddings) if cn in nonempty_documents]
            if filtered_data:
                case_numbers, case_embeddings = zip(*filtered_data)
                case_numbers = list(case_numbers)
                case_embeddings = list(case_embeddings)
            else:
                st.error("No embeddings found for nonempty documents!")
                return
    else:
        with st.spinner("Computing case embeddings..."):
            case_embeddings = embedding_model.encode(case_texts, show_progress_bar=True)
            case_numbers = list(nonempty_documents.keys())
    
    # User query input for case retrieval
    query = st.text_area("Enter your legal query:", placeholder="E.g., 'Which cases relate to patent infringement?'")
    
    if st.button("Search Cases"):
        if not query.strip():
            st.error("Please enter a query.")
            return
            
        with st.spinner("Finding relevant cases..."):
            similar_cases, similarity_scores = find_most_similar_cases(
                query, embedding_model, case_embeddings, case_numbers, k=top_k
            )
        
        st.subheader(f"Top {top_k} Most Relevant Cases")
        
        # Debug: Print similar cases and their scores
        print("Debug: Similar cases found:", similar_cases)
        print("Debug: Similarity scores:", similarity_scores)
        
        # Display first few lines for each similar case
        for case_num in similar_cases:
            if case_num not in nonempty_documents:
                print(f"Debug: Case {case_num} not found in nonempty_documents keys: {list(nonempty_documents.keys())}")
            else:
                document_text = nonempty_documents[case_num]
                print(f"Debug: Document for {case_num} has length {len(document_text)}")
                lines = document_text.splitlines()
                print(f"First few lines for case {case_num}:")
                for line in lines[:10]:
                    print(line)
                print("\n" + "-" * 40 + "\n")
        
        # Generate summaries in parallel for the similar cases found
        summaries = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_case = {
                executor.submit(safe_summarize_first_part_by_tokens, nonempty_documents[case_num], summarizer, 1024): case_num
                for case_num in similar_cases if case_num in nonempty_documents
            }
            for future in concurrent.futures.as_completed(future_to_case):
                case_num = future_to_case[future]
                try:
                    result = future.result()
                    print(f"Debug: Summary for case {case_num}: {result[:100]}")
                    summaries[case_num] = result
                except Exception as exc:
                    summaries[case_num] = f"⚠️ Failed to summarize: {exc}"
                    print(f"Debug: Exception for case {case_num}:", exc)


        # Display each of the top similar cases with their summaries and excerpts
        for i, (case_num, score) in enumerate(zip(similar_cases, similarity_scores[:len(similar_cases)]), 1):
            if case_num not in nonempty_documents:
                continue
            with st.expander(f"Case #{i}: {case_num} (Similarity: {score:.2f})"):
                url = get_case_url(case_num)
                st.markdown(f"[View full case on Justia]({url})")
                st.subheader("Summary")
                st.write(summaries.get(case_num, "No summary available."))
                st.subheader("Excerpt")
                excerpt = nonempty_documents[case_num][:500] + "..." if len(nonempty_documents[case_num]) > 500 else nonempty_documents[case_num]
                st.text(excerpt)
        
        # # Generate response for the top-most similar case
        # st.subheader("Generated Response for Top Case")
        # with st.spinner("Generating response for the top case..."):
        #     top_case = similar_cases[0]
        #     top_case_text = nonempty_documents[top_case]
        #     generated_response = generate_response_for_case(top_case_text, chunk_size=200, overlap=50)
        # st.write(generated_response)
        
        # # if show_visualization:
        # #     st.subheader("Case Embeddings Visualization")
        # #     fig = visualize_embeddings(case_embeddings, case_numbers)
        # #     st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
