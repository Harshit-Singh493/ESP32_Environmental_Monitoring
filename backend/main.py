from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@app.get('/nigga')
def hello():
    return "Hello"