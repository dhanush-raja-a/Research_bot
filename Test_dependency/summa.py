import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
import faiss
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import streamlit as st

st.title("Document-based Question Answering ")
st.write("Rag bassed question answering system")
#load the dotenv
from dotenv import load_dotenv
load_dotenv()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
#loading the llm model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=2,
    # other params...
)
#loading the file
file_path="/Users/dhanushrajaa/Desktop/Reserach_Chat_Bot/data/A_Study_on_the_Application_of_TensorFlow_Compression_Techniques_to_Human_Activity_Recognition.pdf"
loader=PyPDFLoader(file_path)
doc=loader.load()
#splitting the file
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(doc)
#intiating the sentence transformer model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#storing in database
faiss_db = FAISS.from_texts([t.page_content for t in all_splits], embedding_model)
faiss_db.save_local("faiss_index")
print("Data successfully stored in Faiss!")
#loadiing form the database
faiss_db = FAISS.load_local("faiss_index", embedding_model,allow_dangerous_deserialization=True)



system_prompt=(
    "you are an assistant for question-aswering tasks,"
    "use the following pieces of retrieved context to answer"
    "the questio if you dot know the answer, say that you"
    "don't know use the three sentences maximum and keep the answe concise"
    "\n\n"
    "{context}"

) 

Prompt= ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)
retriever = faiss_db.as_retriever()
question_answer_chain=create_stuff_documents_chain(llm, Prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)
msg=st.text_area("Enter your question here:")
response=rag_chain.invoke({"input": msg})
# print("response :", response["answer"])
st.write(response["answer"])