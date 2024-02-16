from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str = Field(description="Azure OpenAI API Key")
    AZURE_OPENAI_API_ENDPOINT: str = Field(description="Azure OpenAI Endpoint")
    AZURE_OPENAI_API_VERSION: str = Field(description="Azure OpenAI API Version")
    AZURE_DEPLOYMENT_NAME: str = Field(
        "gpt-35-turbo", description="Azure Deployment Name"
    )
    AZURE_DEPLOYMENT_MODEL: str = Field(
        "gpt-35-turbo", description="Azure Deployment Model"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
