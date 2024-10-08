from pathlib import Path
from dotenv import load_dotenv

SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent

load_dotenv()