# RAG-based Question Answering System
---

## Overview
The Retriever-Augmented Generation (RAG) project aims to enhance question-answering capabilities by combining the strengths of a Language Model (LLM) with a document retriever system using FAISS (Facebook AI Similarity Search). This project is built using a variety of libraries including Streamlit, LangChain, PyPDF2, and dotenv. It is designed to handle user queries based on context provided from uploaded documents, particularly PDFs.

## Features
- **Document Upload and Text Extraction:** Users can upload PDF documents from which text is extracted and used as context.
- **Text and Code Descriptions:** Provides detailed explanations and examples for input text or code, making it accessible to users of all knowledge levels.
- **Retrieval-Augmented Response Generation:** Integrates a retrieval system with an LLM to provide context-aware answers based on the text extracted from uploaded documents.
- **Local Storage of Context:** Uses FAISS to create and query vector embeddings of the extracted text for efficient retrieval.

## Installation
1. **Clone the Repository:**
   ```
   git clone [repository-url]
   cd [repository-folder]
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY='your_openai_api_key_here'
   ```

4. **Run the Application:**
   ```
   streamlit run app.py
   ```

## Usage
- **Starting the Application:** Open your terminal, navigate to the project directory, and run `streamlit run app.py`.
- **Interacting with the Application:** 
  - **Upload a PDF:** Users can upload a PDF file for which they want to get explanations or have questions answered.
  - **Enter a Query:** Type your query into the text input field and click "Submit" to receive an answer based on the uploaded document's context.

## Code Explanation
- **Dependencies Import:** Imports necessary libraries and modules required for the application.
- **FAISS Vector Store Creation:** Extracts text from PDFs, splits the text into manageable chunks, and converts these chunks into vector embeddings stored in a FAISS index.
- **RAG Function:** Combines a retriever (FAISS) and a language model to provide answers to user queries. It loads vector embeddings, retrieves relevant documents, and generates responses using the LLM.
- **Streamlit UI:** Sets up a user-friendly interface with Streamlit for uploading documents, entering queries, and displaying responses.

## Contribution
Contributions to the RAG project are welcome. Please ensure to follow the best practices for coding and documentation. Submit pull requests for any enhancements.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

---


