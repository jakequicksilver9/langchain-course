import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredLoader(file_path="C:/Projs/langchain-course/mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
