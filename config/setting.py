import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 1. 大模型 API 配置 (豆包/OpenAI)
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str
    
    # 2. 外部工具配置 (Exa 搜索)
    EXA_API_KEY: str

    # 3. LangSmith 监控追踪配置

    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_ENDPOINT: str
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str

    model_config = SettingsConfigDict(
        env_file=".env.example", 
        env_file_encoding="utf-8",
        extra="ignore" # 如果 .env 里有多余的变量，忽略它们不报错
    )

# 实例化这个类，以后全项目导入这个 settings 就行了
settings = Settings()

if settings.LANGCHAIN_TRACING_V2.lower() == "true":
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ["LANGSMITH_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGCHAIN_PROJECT