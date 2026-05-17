# Production-Ready GenAI & RAG Engine
Vídeo demostrativo:


https://github.com/user-attachments/assets/556a3376-a0df-4e74-a0d9-a48572649a89



An advanced, production-ready Generative AI and Retrieval-Augmented Generation (RAG) backend engine designed for semantic document answering, entity extraction, and structured text summarization. Built using modern cloud-native architectures, high-performance API frameworks, and standard AI engineering paradigms.

## 🚀 Key Features & Responsibilities Addressed

*   **RAG Pipeline Engineering**: Complete end-to-end data ingestion utilizing `RecursiveCharacterTextSplitter` for optimal context chunking and semantic preservation.
*   **Vector Database Integration**: Native orchestration with dense retrieval vector indexers (`FAISS`) using state-of-the-art multi-dimensional embeddings (`text-embedding-3-small`).
*   **Production REST APIs**: High-throughput asynchronous endpoints powered by `FastAPI`, following strict enterprise design patterns, dynamic query data-validation models (`Pydantic`), and native multipart stream handling.
*   **Advanced Prompt Engineering**: LLM-agnostic chaining pipelines (`LangChain LCEL`) combined with decoupled chat prompt orchestration to strictly bound LLM hallucination limits.
*   **Cloud & DevOps Ready**: Pre-configured multi-stage `Dockerfile` optimization for rapid container microservice horizontal deployment on **AWS (ECS/EKS)** or **Azure App Services**.

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **Frameworks**: FastAPI, LangChain, LangChain-Community, LangChain-OpenAI
*   **Vector Stores**: FAISS (Facebook AI Similarity Search)
*   **Models**: OpenAI (GPT-4o-mini), text-embedding-3-small
*   **Environment**: Docker, Uvicorn, Pydantic v2

## 📦 Architecture & Directory Structure

```text
📦 genai-rag-assistant
 ┣ 📂 app
 ┃ ┣ 📂 api         # REST API routes and endpoint validation
 ┃ ┣ 📂 core        # Settings, configurations, and environment loaders
 ┃ ┣ 📂 services    # Core AI Logic (LLM bindings, retrievers, prompt chains)
 ┃ ┗ 📜 main.py     # Application bootstrap and entry point
 ┣ 📂 data          # Sandbox local data for validation and data ingestion
 ┣ 📂 notebooks     # Experimental LangChain / LangGraph rapid testing environments
 ┣ 📜 .env.example  # Global environment variables blueprint
 ┣ 📜 Dockerfile    # Standard OCI-compliant container configuration
 ┗ 📜 requirements.txt
```

## ⚡ Getting Started Locally

### 1. Environment Setup
Clone the repository and make sure your virtual environment is initialized:
```bash
git clone https://github.com
cd genai-rag-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Configure your environmental secrets. Duplicate the blueprint and fill in your keys:
```bash
cp .env.example .env
# Edit .env file and paste your OPENAI_API_KEY
```

### 3. Execution
Fire up the asynchronous local development server:
```bash
uvicorn app.main:app --reload
```
Access the interactive OpenAPI Documentation (Swagger UI) at: `http://127.0.0`

## 🐳 Containerized Deployment (AWS / Azure)

Build and run your isolated cloud-native microservice instantly using Docker:
```bash
# Build target image
docker build -t genai-rag-assistant:latest .

# Run mapping to external network interfaces
docker run -d -p 8000:8000 --env-file .env genai-rag-assistant:latest
```
This artifact is prepared for seamless integration into AWS ECS/Fargate task definitions or Azure Container Instances (ACI).
