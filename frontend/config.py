import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


load_dotenv()


def get_api_url() -> str:
    env_api_url = os.getenv("API_URL")
    if env_api_url:
        return env_api_url

    local_secrets_files = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    if not any(path.exists() for path in local_secrets_files):
        return "http://localhost:8000"

    try:
        return st.secrets["API_URL"]
    except (FileNotFoundError, KeyError):
        return "http://localhost:8000"
