from fastapi import FastAPI
from os.path import join, dirname, realpath

app = FastAPI()
UPLOADS_PATH = join(dirname(realpath(__file__)), 'static/uploads/')

from speech_service.speech_main import *


@app.get("/")
async def root():
    return {"message": "Hello World"}
