from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

if __name__ == "__main__":

    print("Loading Documents...")
    loader = TextLoader("./information.txt")
    document = loader.load()
    print(f"Loaded {len(document)} documents")

    print("Splitting Documents")
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_documents = splitter.split_documents(document)
    print(f"splitted into {len(split_documents)} chunks")

    print("Started Embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    print("Inserting into VectorDB")
    vector_db = PineconeVectorStore.from_documents(split_documents, embeddings, index_name="project")
    print(f"Inserted {len(split_documents)} documents in the VectorDB")
