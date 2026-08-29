import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma_db"

def load_indian_food_dataset():
    print("Loading IndianFoodDatasetCSV.csv...")
    df = pd.read_csv("data/IndianFoodDatasetCSV.csv")
    documents = []
    for _, row in df.iterrows():
        name = str(row.get("TranslatedRecipeName", "")).strip()
        ingredients = str(row.get("TranslatedIngredients", "")).strip()
        instructions = str(row.get("TranslatedInstructions", "")).strip()
        cuisine = str(row.get("Cuisine", "")).strip()
        course = str(row.get("Course", "")).strip()
        diet = str(row.get("Diet", "")).strip()
        prep_time = str(row.get("PrepTimeInMins", "")).strip()

        if not name or not instructions:
            continue

        content = f"""Recipe: {name}
Cuisine: {cuisine}
Course: {course}
Diet: {diet}
Prep Time: {prep_time} mins

Ingredients:
{ingredients}

Instructions:
{instructions}"""

        documents.append(Document(
            page_content=content,
            metadata={"recipe_name": name, "cuisine": cuisine, "source": "IndianFoodDatasetCSV"}
        ))
    print(f"  -> {len(documents)} documents")
    return documents


def build_chroma():
    all_documents = load_indian_food_dataset()
    print(f"\nTotal documents: {len(all_documents)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    chunks = splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    batch_size = 500
    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if db is None:
            db = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_PATH)
        else:
            db.add_documents(batch)
        print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"Done! Vector DB saved to '{CHROMA_PATH}'.")
    return db


if __name__ == "__main__":
    build_chroma()