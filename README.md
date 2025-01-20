# RAG-based Question Answering System

This project implements a Retrieval-Augmented Generation (RAG) system using OpenAI's GPT-3.5-turbo model and FAISS for vector-based similarity search. The system can handle both user-provided text input and PDF documents for question answering. 

## Features

- **Text Input**: Accepts user-provided text or code for explanation and generates detailed, easy-to-understand answers.
- **PDF Support**: Allows users to upload a PDF file (e.g., a resume) for processing and answering questions based on its content.
- **Chunking and Vectorization**: Splits large documents into manageable chunks and stores them in a FAISS vector store for efficient retrieval.
- **Customizable Prompting**: Uses advanced prompting techniques to guide the model in generating contextually accurate and concise responses.
- **Interactive Web App**: Built with Streamlit for a user-friendly interface.

## How It Works

1. **File Upload**: Users can upload a PDF document.
2. **Text Extraction**: The system extracts text from the uploaded PDF using `PyPDF2`.
3. **Chunking**: The extracted text is split into chunks using `RecursiveCharacterTextSplitter`.
4. **Vector Store**: The chunks are embedded using `OpenAIEmbeddings` and stored in a FAISS vector store for retrieval.
5. **Query Processing**:
    - If no PDF is uploaded, the system directly processes the user’s text input using a description chain.
    - If a PDF is uploaded, the system retrieves the most relevant chunks from the vector store and generates a response using the LLM.
6. **Question Answering**: The system leverages a retrieval-augmented generation chain (`create_retrieval_chain`) to provide precise and contextually relevant answers.

## Project Structure

