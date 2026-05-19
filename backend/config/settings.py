from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    DATABASE_URL: str = Field(default="sqlite:///./alloy_iq.db", env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    SECRET_KEY: str = Field(default="change-me-in-production-please", env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TTL_MIN: int = Field(default=60 * 24, env="ACCESS_TTL_MIN")  # 24 hours
    BLENDER_PATH: str = Field(default="blender", env="BLENDER_PATH")
    RENDERS_DIR: str = Field(default="frontend/public/renders", env="RENDERS_DIR")
    
    # Materials Project & AFLOW integration
    MP_API_KEY: Optional[str] = Field(default=None, env="MP_API_KEY")
    AFLOW_BASE_URL: str = Field(default="http://aflow.org/API/aflowlib.org", env="AFLOW_BASE_URL")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_DIR: str = Field(default="logs", env="LOG_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
