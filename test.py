from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from together import Together
import random

app = FastAPI()
client = Together()

@app.get("/generate_question/{request}")
async def generate_coding_question(request):
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=[
                {"role": "system", "content": "You are an AI that generates coding questions."},
                {"role": "user", "content": f"Generate only one random and unique coding question about {request} and just return the question in string format."},
            ],
            temperature=round(random.uniform(0.4, 0.9), 1),
        )
        question_content = response.choices[0].message.content
        return (question_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
