import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

INPUT_PATH = os.getcwd()
OUTPUT_PATH = os.path.join(os.getcwd(), "figures")
DATA_FOLDER = os.path.join(os.getcwd(), "data")
SLIDES_OUTPUT_PATH = os.path.join(os.getcwd(), "slides")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(SLIDES_OUTPUT_PATH, exist_ok=True)