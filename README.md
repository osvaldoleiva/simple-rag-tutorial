# Local RAG Chatbot with Ollama

This project is a local implementation of a Retrieval-Augmented Generation (RAG) chatbot, originally based on the tutorial from [Hugging Face](https://huggingface.co/blog/ngxson/make-your-own-rag). The implementation has been modified to run entirely locally using Ollama for model inference.

## Features

- 🚀 **Fully Local**: No API keys or internet connection required after initial setup
- 🐱 **Cat Facts Knowledge Base**: Comes pre-loaded with interesting cat facts
- 🔍 **Semantic Search**: Finds relevant information using vector similarity
- 💬 **Interactive Chat**: Ask questions and get answers based on the knowledge base

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required Python packages (install via `pip install -r requirements.txt`)

## Setup

1. **Install Ollama**
   Download and install Ollama from [ollama.ai](https://ollama.ai/)

2. **Start the Ollama server**
   ```bash
   ollama serve
   ```

3. **Pull required models** (in a new terminal):
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the chatbot:
   ```bash
   python main.py
   ```

2. When prompted, type your question about cats and press Enter

3. Type 'quit', 'exit', or 'q' to exit the program

## How It Works

1. **Data Loading**: The script loads cat facts from `cat-facts.txt`
2. **Embedding Generation**: Each fact is converted into a vector embedding using `nomic-embed-text`
3. **Query Processing**: When you ask a question:
   - The question is converted to an embedding
   - The system finds the most similar facts using cosine similarity
   - The relevant context is sent to the language model (`llama3`)
   - The model generates a response based on the retrieved context

## Customization

### Using Different Models
You can change the models in `main.py` by modifying these lines:

```python
EMBEDDING_MODEL = 'nomic-embed-text'  # Other options: 'all-minilm', 'bge-small', etc.
LANGUAGE_MODEL = 'llama3'  # Other options: 'mistral', 'llama2', etc.
```

### Adding Your Own Knowledge Base
1. Replace `cat-facts.txt` with your own text file
2. Each line should contain a single fact or piece of information
3. The system will automatically process the new file when you run `main.py`

## Troubleshooting

- If you get model not found errors, make sure you've pulled the models with `ollama pull`
- Ensure the Ollama server is running before starting the script
- For large knowledge bases, embedding generation might take some time on the first run

## Credits

Based on the tutorial: [Make Your Own RAG with Hugging Face](https://huggingface.co/blog/ngxson/make-your-own-rag)

## License

This project is open source and available under the MIT License.
