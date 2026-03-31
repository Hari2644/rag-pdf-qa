import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

generator = load_model()

def get_answer(prompt_text, context, query):
    # Just return the most relevant chunk cleanly
    sentences = context.split('. ')
    query_words = query.lower().split()
    scored = []
    for s in sentences:
        score = sum(1 for w in query_words if w in s.lower())
        scored.append((score, s))
    scored.sort(reverse=True)
    top = '. '.join([s for _, s in scored[:3]])
    return top if top else "No clear answer found in the document."

st.title("📄 RAG PDF Q&A System")
st.write("👋 Upload a PDF to start")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.write("Processing PDF...")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    if not documents:
        st.error("PDF could not be read properly.")
    else:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        st.success("✅ PDF processed! Ask your question 👇")

        query = st.text_input("Enter your question:")
        if query:
            docs = db.similarity_search(query)
            if not docs:
                st.write("No relevant info found.")
            else:
                context = " ".join([doc.page_content for doc in docs])
                prompt = f"Answer based on context:\n{context}\n\nQuestion: {query}\nAnswer:"
                answer = get_answer(prompt, context, query)
                st.write("### Answer:")
                st.write(answer)
