from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Carrega chaves do arquivo .env se existir
load_dotenv()

app = FastAPI(
    title="GenAI & RAG Production Engine",
    description="API para motores de resposta, extração de entidades e pipeline RAG.",
    version="1.0.0"
)

# Inicializa o modelo de embeddings (OpenAI)
# Nota: Requer OPENAI_API_KEY configurada no ambiente
if os.getenv("OPENAI_API_KEY"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
else:
    embeddings = None

vector_store = None

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "GenAI RAG API Engine is running smoothly."}

@app.post("/rag/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    """Pipeline RAG: Ingestão de dados, chunking estruturado e criação de embeddings."""
    global vector_store
    if not embeddings:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada no ambiente.")
    
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        # Estratégia de Chunking exigida por engenheiros seniores
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([text])
        
        # Armazenamento e Indexação Vetorial (FAISS)
        if vector_store is None:
            vector_store = FAISS.from_documents(docs, embeddings)
        else:
            vector_store.add_documents(docs)
            
        return {"status": "success", "message": f"Arquivo '{file.filename}' indexado com sucesso no FAISS."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/ask/")
async def answer_question(request: QueryRequest):
    """Answering Engine: Recuperação semântica de contexto e geração de resposta LLM."""
    global vector_store
    if not vector_store:
        raise HTTPException(status_code=400, detail="Nenhum documento foi indexado na base vetorial ainda.")
    
    try:
        # Busca semântica por similaridade (Retriever)
        docs = vector_store.similarity_search(request.question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # LLM e Engenharia de Prompt customizada
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt_template = ChatPromptTemplate.from_template(
            "Você é um assistente especialista em IA. Responda à pergunta do usuário baseando-se estritamente no contexto fornecido.\n\n"
            "Contexto:\n{context}\n\n"
            "Pergunta:\n{question}\n\n"
            "Resposta:"
        )
        
        chain = prompt_template | llm
        response = chain.invoke({"context": context, "question": request.question})
        
        return {"answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/summarize/")
async def summarize_text(text: str):
    """Summarization Engine: Geração de resumos executivos baseados em LLM."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada.")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        prompt = f"Escreva um resumo executivo claro e estruturado em tópicos do seguinte texto:\n\n{text}"
        response = llm.invoke(prompt)
        return {"summary": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
