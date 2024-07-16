# Corrective RAG Implementation

### So, what even is RAG?
When an LLM is asked a question on data it does not have, it will do something called "hallucination". This leads to it making up blatantly wrong facts therefore ruining user expreience. 
Retrieval Augmented Generation helps prevent that by creating what is called a "vector store" for the LLM to search through using a similarity search based on the user's query. Corrective
RAG is a type of RAG which includes web search, scoring, and more

### How do I test this out?
First, ensure you have an OpenAI API key alongside a google.serper.dev API key. Then, fork this repository and follow the following steps:

1. Create a virtual environment
Run the following command in your terminal   
``` cmd
python -m venv .venv
```
2. Install the dependencies onto the virtual environment
Run the following command in your terminal again
``` cmd
pip install -r requirements.txt
```
3. Import your API keys
Create a .env file that looks like this
``` .env
SEARCH_API_KEY= Insert your google.serper.dev api key here
OPEN_AI_KEY= Insert your OpenAI API key here
```
5. Run the program in your terminal using the following command
``` cmd
python main.py
```
