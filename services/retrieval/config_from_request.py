"""Build SearchPipelineConfig from the request body."""

from src.llms.embeddings.openai_embedder import OpenAIEmbedder
from src.retrieval.config import SearchPipelineConfig

from .schemas import SearchRequest


def build_search_pipeline_config(req: SearchRequest) -> SearchPipelineConfig:
    kwargs = dict(
        search_type=req.search_type,
        collection_name=req.collection_name,
        search_limit=req.search_limit,
    )
    embedding_model = getattr(req, "embedding_model", None)
    if embedding_model:
        kwargs["embedder"] = OpenAIEmbedder(
            model=embedding_model,
            count_tokens=False,
        )
    return SearchPipelineConfig(**kwargs)
