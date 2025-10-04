
import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load .env and configure Gemini
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-2.0-flash-lite")

# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "data.csv")
df = pd.read_csv(csv_path)

# Retry-decorated Gemini call
@retry(wait=wait_exponential(min=30, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
def chat_llm(prompt, temperature=0.0, max_tokens=2048):
    """
    Call Gemini API with retry logic
    
    Args:
        prompt: Input prompt
        temperature: Creativity level (0.0-1.0)
        max_tokens: Maximum output tokens (increased default for batch processing)
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini Error: {e}")
        raise
'''import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type



# Load .env and configure Gemini
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "data.csv")
df = pd.read_csv(csv_path)

# Retry-decorated Gemini call
@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
def chat_llm(prompt, temperature=0.0, max_tokens=512):
    try:
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )

        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        raise
'''
