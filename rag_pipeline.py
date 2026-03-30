import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_and_split(file_path: str) -> list:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks: list) -> Chroma:
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )


def vectorstore_exists() -> bool:
    return os.path.exists(CHROMA_PERSIST_DIR) and \
           len(os.listdir(CHROMA_PERSIST_DIR)) > 0


def get_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(llm, retriever):
    prompt = PromptTemplate.from_template("""
You are a helpful AI assistant. Use the following context retrieved from 
uploaded documents to answer the question. If the answer is not in the 
context, say so clearly rather than making something up.

Context:
{context}

Question: {question}

Answer:""")

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ingest_file(uploaded_file) -> tuple:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    chunks = load_and_split(tmp_path)
    os.unlink(tmp_path)
    vectorstore = build_vectorstore(chunks)
    return vectorstore, len(chunks)