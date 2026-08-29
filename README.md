# RecipeRAG


🍲 RecipeRAG

RecipeRAG is an AI-powered food recipe assistant built using LangChain, ChromaDB, Groq LLM, and Streamlit.

The application helps users:

* Search Indian recipes
* Explore ingredients and cooking instructions
* Get nutrition information
* Discover recipes based on ingredients
* Receive food recommendations based on mood, hunger, or activity
* Watch related YouTube recipe videos

Features

* Retrieval-Augmented Generation (RAG)
* ChromaDB vector database
* Groq-powered LLM responses
* Streamlit chat interface
* Recipe recommendation engine
* Nutrition information lookup
* YouTube recipe integration
* Conversation memory and follow-up questions

Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Groq API
* Pandas

Project Structure

RecipeRAG/
│
├── data/
│   ├── IndianFoodDatasetCSV.csv
│   ├── cuisines.csv
│   └── recipe.txt
│
├── ingest.py
├── rag.py
├── streamlit_app.py
├── requirements.txt
├── .env
└── README.md

Installation

git clone https://github.com/riteshnitrkl/RecipeRAG.git
cd RecipeRAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key

Build Vector Database

python ingest.py

Run Application

streamlit run streamlit_app.py

Example Questions

* Tell me about paneer
* Show me chicken recipes
* How to make biryani?
* What should I eat after a workout?
* Nutrition information for almonds
* Suggest a healthy dinner

Author

Ritesh Chaurasiya
