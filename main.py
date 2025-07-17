import os
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
# from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Configuration
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_base = os.getenv("AI_BASE")
openai_api_version = os.getenv("AZURE_OPENAI_VERSION")
deployment_name = os.getenv("AZURE_DEPLOYMENT")

# Create Azure LLM
llm = AzureChatOpenAI(
    api_key=openai_api_key,
    azure_endpoint=openai_api_base,
    deployment_name=deployment_name,
    api_version=openai_api_version,
    temperature=0,
    max_tokens=3000,
)

# Load Excel files
tab1 = "mobility_application_merged_final.xlsx"
tab2 = "portal_application_merged_final.xlsx"

df_mobile = pd.read_excel(tab1)
df_web = pd.read_excel(tab2)
df = pd.concat([df_mobile, df_web], ignore_index=True)

# Clean column names
df.columns = df.columns.str.strip()

# Preprocess documents
def make_documents(df: pd.DataFrame):
    docs = []
    for _, row in df.iterrows():
        try:
            text = (
                f"Application Name: {row.get('Application Name', '')}\n"
                f"Type: {row.get('Application Type', '')}\n"
                f"Criticality: {row.get('Application Criticality (1-5)', '')}\n"
                f"URL: {row.get('Application URL', '')}\n"
                f"Owner: {row.get('Application Owner', '')}\n"
                f"Exposed on Internet: {str(row.get('Application Exposed On Public Internet', '')).strip()}\n"
                f"Users: {row.get('Users (Internal M&M users / External Users / Both)', '')}\n"
                f"Go Live Date (Existing): {row.get('Go Live date for Existing application', '')}\n"
                f"Go Live Date (New): {row.get('Go Live date for Newly created application', '')}\n"
                f"Acunetix Scan Completed: {row.get('Acunetix Scan  Completed? (Yes/No)', '')}\n"
                f"Checkmarx Scan Completed: {row.get('Checkmarx Scan Completed? (Yes/No)', '')}\n"
                f"Manual VAPT Done: {row.get('VAPT Manual Testing Complete? (Yes/No)', '')}\n"
                f"Comments: {row.get('Comments', '')}\n"
            )
            docs.append(text)
        except Exception as e:
            print(f"Row skipped due to error: {e}")
    return docs

docs = make_documents(df)
print(f"Total documents generated: {len(docs)}")

if not docs:
    raise ValueError("No documents were created. Check your data and column names.")

# Embedding and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# embeddings = HuggingFaceEmbeddings(model=model)
vectorstore = FAISS.from_texts(docs, embeddings)

# Build RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})
)


# FastAPI app
app = FastAPI(title="Application Inventory Chatbot")

class Query(BaseModel):
    question: str

# In your FastAPI backend, replace the fixed prompt with a generalized one:
@app.post("/query")
def query_bot(q: Query):
    if not q.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    prompt = (
        "You are a cybersecurity and application‑governance expert with comprehensive access to an evolving inventory of application documents."
        "Whenever the user asks a question, first retrieve all relevant details from the inventory, then:\n"
        "  1. Summarize the raw facts you found (e.g., application names, criticality, scan status, database type, etc.).\n"
        "  2. Provide clear, step‑by‑step reasoning on how you arrived at your conclusions or recommendations.\n"
        "  3. Offer asny insights, patterns, or risks you observe (for instance, \"Applications using SQL databases have the highest exposure risk because...\").\n"
        "Be concise, structured, and use bullet points or numbered lists when appropriate.\n\n"
        f"User Question: {q.question}"
    )

    result = qa.invoke(prompt)

    return {
        "query": q.question,
        "result": result
    }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# Applications used by Internal Mahindra Users

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
