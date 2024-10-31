    # Import necessary libraries
    import os
    import gradio as gr
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_ollama import OllamaLLM
    from sentence_transformers import SentenceTransformer
    from langchain.embeddings.base import Embeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.prompts import PromptTemplate

    # Custom embedding class
    class SentenceTransformerEmbeddings(Embeddings):
        def __init__(self, model_name="all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)

        def embed_documents(self, texts):
            return self.model.encode(texts).tolist()

        def embed_query(self, text):
            return self.model.encode([text]).tolist()[0]

    # Directory containing the PDF files
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")

    # Load and parse all PDFs in the "docs" directory
    all_documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(docs_dir, filename)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)

    # Split documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    texts = [chunk for doc in all_documents for chunk in text_splitter.split_text(doc.page_content)]

    # Initialize the custom embeddings model
    embedding_function = SentenceTransformerEmbeddings()

    # Create a FAISS vector store with texts and custom embedding function
    docsearch = FAISS.from_texts(texts, embedding_function)

    prompt_template = """Use o seguinte contexto para responder à pergunta. Se a resposta não estiver contida no contexto, diga "Eu não sei".

    Contexto: {context}

    Pergunta: {query}"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

    # Initialize the Ollama model with optimized settings
    ollama_model = OllamaLLM(model="llama3.2", max_tokens=1000)

    # Criando a Cadeia de Perguntas e Respostas com prompt personalizado
    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama_model,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 5}), # k=3
        chain_type_kwargs={"prompt": PROMPT}, # Adicionando prompt
    )

    # Set up the retriever to ensure responses come from the PDF content only
    retriever = docsearch.as_retriever(search_kwargs={"k": 1})

    # Set up the RetrievalQA chain without a custom prompt or document variable name
    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False  # No need for source documents in the output
    )

    # Define the function for Gradio to process the input query and generate an answer
    def chat_with_pdf(query):
        # Pass `query` directly to the chain and extract the answer
        answer = qa_chain.run(query)
        return f"Resposta: {answer}"

    # Set up the Gradio interface
    iface = gr.Interface(
        fn=chat_with_pdf,
        inputs="text",
        outputs="text",
        title="Chatbot - Engenharia da Computação",
        description="Pergunte qualquer coisa com base no conteúdo dos arquivos PDF localizados no diretório 'docs'."
    )

    # Launch the interface
    iface.launch()
