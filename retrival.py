from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    llm = ChatGoogleGenerativeAI( model='gemini-1.5-flash')
    
    query = "Tell me about 3 places to visit in India"

    vectorstore = PineconeVectorStore(index_name="project",embedding=embeddings)

    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combined_docs_chain = create_stuff_documents_chain(llm,prompt)

    retriever_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),
    combine_docs_chain=combined_docs_chain)

    result = retriever_chain.invoke({'input': query})
    print(result)

