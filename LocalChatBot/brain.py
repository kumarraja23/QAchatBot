from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
import time

temp_dir = r"D:\Data\Official\PDf_qa\Uploaded_docs"


def save_file(uploaded_file):
    start_time = time.time()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    print("FILE SAVED HERE", path)
    end_time = time.time()
    print(f"Time taken by saving the file: {end_time - start_time:.2f} seconds")
    return path

def upload_pdf(pdf_path):
    # loader = UnstructuredPDFLoader(file_path = pdf_path)
    start_time = time.time()
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    print("uploaded file is loaded")
    end_time = time.time()
    print(f"Time taken by data parsing the PDF: {end_time - start_time:.2f} seconds")
    return pages



def contentLoader(urls):
    start_time = time.time()
    listOfUrls = urls.split("\n")
    pages = [WebBaseLoader(url).load() for url in listOfUrls]
    docs = [item for sublist in pages for item in sublist]
    # load_and_save_content(listOfUrls, "webscraping.txt")
    with open('webscraping.txt', 'w', encoding='utf-8') as f:
        for sublist in pages:
            for item in sublist:
                f.write(item.page_content)
                f.write("\n")
        print(f"Successfully saved content")
    
    print("URLs are loaded")
    end_time = time.time()
    print(f"Time taken by contentLoader from URL's: {end_time - start_time:.2f} seconds")
    return docs

def split_data_into_chunks(data):
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    print("Text is split into chunks")
    end_time = time.time()
    print(f"Time taken by spliting the data into chunks: {end_time - start_time:.2f} seconds")
    return chunks

def vectorDataBase(chunks):
    start_time = time.time()

    vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag" )
    print("data is stored in vectorDB")

    end_time = time.time()
    print(f"Time taken by the data to stored in vector Data Base: {end_time - start_time:.2f} seconds")
    return vector_db

def initialize_conversation_memory(memory_key="chat history"):
    memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)
    print("Memory is initialized")
    return memory

def initialize_the_model(model_name, temp):
    start_time = time.time()
    print(f"Selected model is {model_name} and Temperature is {temp}")
    local_model = ChatOllama(model = model_name, temperature=temp)

    prompt_template = """Answer the question based only on the following context:
    {context}
    Question: {question}.
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    print("Model Is Initialized")

    end_time = time.time()
    print(f"Time taken to initialize the model: {end_time - start_time:.2f} seconds")
    return local_model, prompt

def retriver_of_Data(retriever, llm):
    start_time = time.time()
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],

    template="""You are an AI language model assistant named as IndChatBot Powered by Indium Software. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",)

    retrievedData = MultiQueryRetriever.from_llm(
    retriever, 
    llm,
    prompt=QUERY_PROMPT)
    end_time = time.time()
    print(f"Time taken by retriver_of_Data: {end_time - start_time:.2f} seconds")
    return retrievedData

def retrieval_qa_chain(retrievedData,llm,prompt):
    start_time = time.time()
    # retriever = vectorDataBase(chunks).as_retriever()
    # qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, memory=memory, chain_type_kwargs={"prompt": prompt})
    qa_chain = (
        {"context": retrievedData, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Chain")
    end_time = time.time()
    print(f"Time taken by retrieval_qa_chain: {end_time - start_time:.2f} seconds")
    return qa_chain

def generate_answer(question, qa_chain):
    start_time = time.time()
    # result = qa_chain({"query": question})
    result = qa_chain.invoke(question)
    print("Answer is generated!!")
    end_time = time.time()
    print(f"Time taken by generate_answer: {end_time - start_time:.2f} seconds")
    return result

