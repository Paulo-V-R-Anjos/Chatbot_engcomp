# PDF-based Chatbot using LangChain and LLaMA

This project implements a chatbot that answers questions based on the content of PDF documents loaded from a specified directory. It utilizes LangChain for handling document retrieval and querying with LLaMA as the language model.

## Features

- Processes PDF files in a specified folder (`docs`) to create a question-answering system.
- Uses FAISS for vector storage and efficient retrieval of document embeddings.
- Leverages the LLaMA model (via Ollama) for generating responses based on retrieved document content.

## Setup Instructions

### Prerequisites

Ensure you have Python 3.7+ installed.

### Step 1: Install Dependencies

1. Clone this repository and navigate to the project directory:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Pull the LLaMA Model

Since this project relies on a LLaMA model, you need to pull the model using Ollama. Run the following command to pull the model (ensure you have Ollama CLI installed):
   ```bash
   ollama pull llama3.2
   ```

### Step 3: Prepare PDF Documents
Add any PDF documents you want the chatbot to use in the docs folder at the same level as main.py. The system will process all PDFs in this folder.

### Step 4: Run the Chatbot
To start the chatbot, run:
    ```python main.py```