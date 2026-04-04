"""CLI for generating answers from retrieval traces."""

import json # للتعامل مع ملفات JSON (قراءة/كتابة)
import logging  # لتسجيل الأحداث (logs)
import os # للوصول إلى متغيرات البيئة
from pathlib import Path # للتعامل مع المسارات بشكل نظيف

import click # مكتبة لإنشاء CLI بسهولة
from dotenv import load_dotenv # لتحميل متغيرات البيئة من ملف .env
# استيراد مكونات المشروع
from corag.controller.base import GenerationConfig  # إعدادات التوليد (مثل temperature)
from corag.controller.openai_controller import OpenAIController # Controller للتعامل مع OpenAI
from corag.corpus.document import Chunk # يمثل جزء من document
from corag.generation.synthesizer import Synthesizer # المسؤول عن توليد الإجابة
from corag.retrieval.state import RetrievalState, RetrievalStep # تمثيل حالة الاسترجاع

load_dotenv() # تحميل متغيرات البيئة (مثل API KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--trace", required=True, type=click.Path(exists=True), help="Trace JSON file"
)
@click.option("--output", required=True, type=click.Path(), help="Output markdown file")
@click.option("--provider", default="openai", help="LLM provider")
@click.option(
    "--model",
    default=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
    help="LLM model",
)
@click.option("--temperature", default=0.3, type=float, help="Temperature")
def main(
    trace: str,
    output: str,
    provider: str,
    model: str,
    temperature: float,
) -> None:
    """Generate answer from retrieval trace."""
    trace_path = Path(trace)
    output_path = Path(output)

    # Load trace
    logger.info(f"Loading trace from {trace_path}")
    with open(trace_path) as f:
        trace_data = json.load(f)

    # Reconstruct state
    state = RetrievalState(original_question=trace_data["original_question"])
    state.executed_queries = trace_data["executed_queries"]
    state.answered_aspects = trace_data["answered_aspects"]
    state.gaps = trace_data["gaps"]
    state.contradictions = trace_data["contradictions"]
    state.is_sufficient = trace_data["is_sufficient"]

    for step_data in trace_data["steps"]:
        chunks = [Chunk.from_dict(c) for c in step_data["chunks"]]
        step = RetrievalStep(
            step_num=step_data["step_num"],
            query=step_data["query"],
            chunks=chunks,
            scores=step_data["scores"],
            rationale=step_data["rationale"],
            metadata=step_data.get("metadata", {}),
        )
        state.steps.append(step)

    # Initialize controller
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        controller = OpenAIController(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Create synthesizer
    config = GenerationConfig(temperature=temperature)
    synthesizer = Synthesizer(controller=controller, config=config)

    # Generate answer
    logger.info("Synthesizing answer...")
    answer, citations = synthesizer.synthesize(state)

    # Format with citations
    full_answer = synthesizer.format_answer_with_citations(answer, citations)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Question\n\n{state.original_question}\n\n")
        f.write(f"# Answer\n\n{full_answer}\n")

    logger.info(f"Saved answer to {output_path}")
    print(f"\n{full_answer}")


if __name__ == "__main__":
    main()
