from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Load all-MiniLM model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
def dynamic_chunker(text: str, C_min=400, C_max=2000, alpha=30, beta=0.20):
    N = len(text)

    # Formula-based chunk size
    chunk_size = min(C_max, max(C_min, int(alpha * (N ** 0.6))))
    overlap = int(beta * chunk_size)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)
    return chunks, chunk_size, overlap

# Example

chunks, size, overlap = dynamic_chunker(contract_document)

print(f"Doc length: {len(contract_document)}")
print(f"Chunk size: {size}, Overlap: {overlap}, Total chunks: {len(chunks)}")
print(chunks[0][:200])

from langchain.vectorstores import Chroma

# Create vectorstore from text chunks
vectordb = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="contract_db"
)

# Persist if you want
vectordb.persist()
# Create a retriever
retriever = vectordb.as_retriever(
    search_type="similarity",   # could also use "mmr"
    search_kwargs={"k": 5}      # top 5 relevant chunks
)

# Example query
query = "payment terms and deadlines"
results = retriever.get_relevant_documents(query)

# Inspect results
for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print("Text:", doc.page_content[:300])
    if hasattr(doc, "metadata"):
        print("Metadata:", doc.metadata)
    print("---")