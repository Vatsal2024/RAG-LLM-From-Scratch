# RAG-LLM-From-Scratch

This repository hosts an implementation of Retrieval-Augmented Generation (RAG) using a Language Model (LLM) specifically designed for querying PDF documents from scratch. Unlike existing solutions that rely on high-level frameworks, this implementation is built ground-up, offering a comprehensive approach to extracting insights from PDF files.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Vatsal2024/RAG-LLM-From-Scratch.git
    cd RAG-LLM-From-Scratch
    ```

2. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Login to Hugging Face using the CLI to access the pre-trained LLM models:

    ```sh
    !huggingface-cli login
    ```

    Follow the instructions to complete the login process.

4. Access the required LLM models from Hugging Face by specifying them in the `main.py` file.

## Usage

1. Place your PDF file in the root directory.

2. Update the `pdf_path` variable in `main.py` with the path to your PDF file.

3. Specify your query in `main.py`.

4. Run the script:

    ```sh
    python main.py
    ```

5. Follow the prompts to input the PDF path and query, and obtain insightful answers from the document.

## Accessing LLMs from Hugging Face

Hugging Face provides a wide range of pre-trained language models, including LLMs, which can be accessed via their model hub. By logging in to Hugging Face using the CLI, you gain access to these models, enabling you to download and utilize them in your projects.
