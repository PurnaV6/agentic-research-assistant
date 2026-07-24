import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))
MAX_SEARCH_RESULTS = 5
MAX_PAGE_CHARS = 4000
