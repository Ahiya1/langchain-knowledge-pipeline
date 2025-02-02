import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pinecone import Pinecone
from openai import OpenAI

class LangChainDocsConfig(BaseModel):
    """Configuration for the LangChain Documentation Pipeline"""
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    indexes: List[str] = Field(
        default=[
            "langchain-langchain",
            "langchain-langsmith",
            "langchain-langgraph"
        ],
        description="List of Pinecone indexes to search"
    )

class LangChainDocsPipeline:
    def __init__(self, config: Dict):
        self.config = LangChainDocsConfig(**config)
        self.pc = Pinecone(api_key=self.config.pinecone_api_key)
        self.client = OpenAI(api_key=self.config.openai_api_key)
        self.indexes = {name: self.pc.Index(name) for name in self.config.indexes}
    
    def get_embeddings(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536
        )
        return response.data[0].embedding
    
    def query_indexes(self, query_embedding: List[float]) -> List[Dict]:
        all_results = []
        for index_name, index in self.indexes.items():
            results = index.query(
                vector=query_embedding,
                top_k=self.config.top_k,
                include_metadata=True
            )
            for match in results.matches:
                match.metadata["source_index"] = index_name
                all_results.append(match)
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:self.config.top_k]
    
    def format_context(self, results: List[Dict]) -> str:
        context_parts = []
        for result in results:
            metadata = result.metadata
            source = metadata.get("source_index", "").replace("langchain-", "")
            url = metadata.get("url", "")
            text = metadata.get("text", "")
            context_parts.append(
                f"Source: {source}\n"
                f"URL: {url}\n"
                f"Content:\n{text}\n"
                f"---\n"
            )
        return "\n".join(context_parts)

    async def __call__(self, messages: List[Dict], config: Optional[Dict] = None) -> Dict:
        user_message = messages[-1]["content"]
        try:
            query_embedding = self.get_embeddings(user_message)
            results = self.query_indexes(query_embedding)
            context = self.format_context(results)
            
            prompt = f"""You are Claude, an AI assistant with access to LangChain, LangSmith and LangGraph documentation. 
Use the following context to answer questions. If you cannot answer based on the context, say so.

Context:
{context}

Remember to:
1. Only use information from the provided context
2. Include relevant URLs
3. Acknowledge if you cannot answer from the given context
4. Format code examples using markdown"""
            
            return {
                "prompt": prompt,
                "context_sources": [
                    {
                        "url": r.metadata.get("url"),
                        "source": r.metadata.get("source_index"),
                        "score": r.score
                    } for r in results
                ]
            }
        except Exception as e:
            return {"prompt": f"Error: {str(e)}", "error": str(e)}

def create_pipeline(pipeline_config: Dict) -> LangChainDocsPipeline:
    return LangChainDocsPipeline(pipeline_config)
