"""CLI for starting the FastAPI server."""

import logging
import os
from pathlib import Path

import click  # لإنشاء CLI (Command Line Interface)
import uvicorn  # لتشغيل FastAPI server
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # لإنشاء API
from pydantic import BaseModel  # لإنشاء نماذج البيانات للتحقق من صحة الطلبات

from corag.controller.base import GenerationConfig
from corag.controller.openai_controller import OpenAIController
from corag.generation.synthesizer import Synthesizer
from corag.indexing.embedder import Embedder
from corag.indexing.index import FAISSIndex
from corag.retrieval.pipeline import RetrievalPipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
app_state: dict[str, object] = {}  # الحالة العامة للتطبيق (تخزين pipeline وsynthesizer)


class AskRequest(BaseModel):
    """Request model for /ask endpoint."""

    question: str
    max_steps: int = 6
    k: int = 8
    temperature: float = 0.2


class AskResponse(BaseModel):
    """Response model for /ask endpoint."""

    question: str
    answer: str
    citations: list[dict]
    num_steps: int
    num_chunks: int


app = FastAPI(

    title="CoRAG API",
    description="Adaptive Multi-Step Retrieval for Complex Queries",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# للتحقق من أن السيرفر يعمل.
@app.get("/healthz")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/version")
async def version() -> dict:
    """Version information."""
    return {"version": "0.1.0", "model": app_state.get("model_name", "unknown")}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a complex question using CoRAG."""
    try:
        # Get components from state
        pipeline: RetrievalPipeline = app_state["pipeline"]  # type: ignore[assignment]
        synthesizer: Synthesizer = app_state["synthesizer"]  # type: ignore[assignment]

        # Update config
        config = GenerationConfig(
            temperature=request.temperature,
            max_tokens=2048,
        )

        # Create temporary pipeline with updated settings
        temp_pipeline = RetrievalPipeline(
            index=pipeline.index,
            embedder=pipeline.embedder,
            controller=pipeline.controller,
            k=request.k,
            max_steps=request.max_steps,
            config=config,
        )

        # Run retrieval
        logger.info(f"Processing question: {request.question}")
        state = temp_pipeline.run(request.question)

        # Generate answer
        answer, citations = synthesizer.synthesize(state)

        return AskResponse(
            question=request.question,
            answer=answer,
            citations=citations,
            num_steps=len(state.steps),
            num_chunks=state.total_chunks_retrieved,
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@click.command()
@click.option(
    "--index-dir", required=True, type=click.Path(exists=True), help="Index directory"
)
@click.option(
    "--embedding-model",
    default="sentence-transformers/msmarco-distilbert-base-v4",
    help="Embedding model",
)
@click.option("--device", default="cpu", help="Device")
@click.option("--provider", default="openai", help="LLM provider")
@click.option(
    "--model", default=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"), help="LLM model"
)
@click.option("--host", default=os.getenv("API_HOST", "0.0.0.0"), help="Host")
@click.option(
    "--port", default=int(os.getenv("API_PORT", "8000")), type=int, help="Port"
)
def main(
    index_dir: str,
    embedding_model: str,
    device: str,
    provider: str,
    model: str,
    host: str,
    port: int,
) -> None:
    """Start the CoRAG API server."""
    index_dir_obj = Path(index_dir)

    # Load components
    logger.info(f"Loading index from {index_dir_obj}")
    index = FAISSIndex.load(index_dir_obj)
    # تحميل قاعدة البيانات ونموذج الـ embeddings.
    logger.info(f"Loading embedder: {embedding_model}")
    embedder = Embedder(model_name=embedding_model, device=device)
    # إعداد مزود الـ LLM (OpenAI) والتحقق من وجود الـ API Key.
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        controller = OpenAIController(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Create pipeline and synthesizer
    config = GenerationConfig(temperature=0.2)
    pipeline = RetrievalPipeline(
        index=index,
        embedder=embedder,
        controller=controller,
        k=8,
        max_steps=6,
        config=config,
    )
    synthesizer = Synthesizer(controller=controller, config=config)

    # Store in app state لتسهيل الوصول إليهم من أي endpoint
    app_state["pipeline"] = pipeline
    app_state["synthesizer"] = synthesizer
    app_state["model_name"] = model

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        app, host=host, port=port
    )  # تشغيل السيرفر باستخدام uvicorn على العنوان والمنفذ المحددين.


if __name__ == "__main__":
    main()

import os
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    with open("interface_template.html", "r") as f:
        content = f.read()
    
    config = f'''{{
        apiKey: "{os.getenv('FIREBASE_API_KEY')}",
        authDomain: "{os.getenv('FIREBASE_AUTH_DOMAIN')}",
        projectId: "{os.getenv('FIREBASE_PROJECT_ID')}",
        storageBucket: "{os.getenv('FIREBASE_STORAGE_BUCKET')}",
        messagingSenderId: "{os.getenv('FIREBASE_MESSAGING_SENDER_ID')}",
        appId: "{os.getenv('FIREBASE_APP_ID')}"
    }}'''
    
    content = content.replace('FIREBASE_CONFIG_PLACEHOLDER', config)
    return content
