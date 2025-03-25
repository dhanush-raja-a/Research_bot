# Research Chat Bot - RAG Question Answering System

A Retrieval Augmented Generation (RAG) based system for answering questions about research papers using GROQ LLM.

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/dhanush-raja-a/Research_bot.git
cd Research_Chat_Bot
```

2. Create and activate virtual environment:

```bash
python -m venv Research_bot
source Research_bot/bin/activate  # For Mac
```

3. Install required packages:

```bash
pip install groq langchain streamlit python-dotenv faiss-cpu sentence-transformers
```

## Configure GROQ API

1. Get your API key:

   - Sign up at [console.groq.com](https://console.groq.com)
   - Navigate to API Keys section
   - Create new API key
   - Copy the key

2. Create .env file:

```bash
touch .env
```

3. Add your API key:

```
GROQ_API_KEY=your_api_key_here  # Replace with actual key
```

## Add Research Papers

1. Place your PDF research papers in the `data` directory:

```python
# Update file path in summa.py:
file_path = "./data/your_research_paper.pdf"  # Replace with your PDF name
```

## Run the Application

```bash
streamlit run Test_dependency/summa.py
```

## Project Structure

```
Research_Chat_Bot/
├── data/                   # Your PDF files go here
├── Test_dependency/
│   └── summa.py          # Main application
├── Research_bot/          # Virtual environment
├── .env                   # API key configuration
└── README.md             # Documentation
```

## Security Notes

- Never commit .env file
- Keep your API key private
- Add .env to .gitignore

## License

MIT License

## Contributing

Pull requests welcome. Please open issues for major changes.
