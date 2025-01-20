import os
import streamlit as st



from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

def get_files(uploaded_file):
    text = ""
    pdf_reader = PdfReader(uploaded_file)  # Pass the uploaded file directly
    for page in pdf_reader.pages:
        if page.extract_text():  # Check if text is extracted
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index", )







def Rag(input):
    # Initialize retriever (FAISS in this case)
    retriever = FAISS.load_local(
        "faiss_index",
        OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
        allow_dangerous_deserialization=True

    ).as_retriever()

    # Initialize LLM
    llm = ChatOpenAI(temperature=0.7,model = 'gpt-3.5-turbo')

    
    system_prompt = """
    You are an intelligent assistant tasked with answering questions based on the provided context. 
    Use the context to generate an accurate and easy-to-understand answer.

    If the input is about code, explain what the code does step-by-step and how it works. 
    If the input is text, provide an explanation that is simple and concise, suitable for anyone to understand. 
    For both text and code, provide one clear and relevant example to illustrate the concept or functionality.

    If the answer is not in the provided context, say: "Answer is not available in the provided file."

    Context:
    {context}


    Response:
    """

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the stuff documents chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    res = chain.invoke({"input": input})

    return res["answer"]




def user_input(question):
    
    response = Rag(question)  
    return response

def desc(input):
    prompt_template = """
    Your task is to provide a clear and easy-to-understand explanation of the given text or code. 
    If the input is a piece of code, explain what the code does step-by-step and how it works. 
    If the input is text, provide an explanation that is simple and concise, suitable for anyone to understand. 
    For both text and code, provide one clear and relevant example to illustrate the concept or functionality.

    If the input is unclear or incomplete, just say, "Explanation not available."

    Input: {text_or_code}
    I want the response in the following format:
    {{
    Explanation:
    Example:
    }}
    """
    description = PromptTemplate(
        input_variables=["text_or_code"],
        template=prompt_template
    )

    result = LLMChain(
        llm=llm,
        prompt=description
    )
    return result.run({"text_or_code": input})

st.set_page_config("P75")

st.title("Ask Me")

col1, col2 = st.columns([6, 1])

with col1:
    text_input = st.text_input("Enter your text", key="text_input")

with col2:
    submit = st.button("Submit")

document = st.file_uploader("Upload Resume", type="pdf", help="please upload the pdf")

if submit:
    if document is None:
        response = desc(text_input)
        st.write(response)
    else:
        doc = get_files(document)  # Pass the uploaded file directly
        chunks = get_chunks(doc)
        vector_store = get_vector_store(chunks)
        response = user_input(text_input)
        st.write(response)
