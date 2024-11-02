from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=360.0)

documents = SimpleDirectoryReader("./data2").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# response = query_engine.query(
#     "What was the total amount of the 2023 Canadian federal budget?"
# )
# print(response)

budget_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="canadian_budget_2023",
    description="A RAG engine with some basic facts about the 2023 Canadian federal budget.",
)


agent = ReActAgent.from_tools(
    [budget_tool], verbose=True
)

response = agent.chat(
    "What is the total amount of the 2023 Canadian federal budget multiplied by 3? Go step by step, using a tool to do any math."
)

print(response)
