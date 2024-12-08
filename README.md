# Basic RAG Using DSPy

A simple implementation of Retrieval-Augmented Generation (RAG) using DSPy framework.

## Setup

1. Clone the repository:
```bash
git clone git@github.com:saeedhumai/basic_rag_using_dspy.git
cd basic_rag_using_dspy
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install dspy-ai openai python-dotenv
```

4. Create `.env` file and add your OpenAI API key:
```bash
OPENAI_API_KEY=your-api-key-here
```

## Usage

Run the main script:
```bash
python ragdspy.py
```

## Features

- Uses DSPy framework for RAG implementation
- Integrates with OpenAI's GPT-4 model
- Environment variable management for API keys
- Simple question-answering interface 