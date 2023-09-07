from fastapi import FastAPI
from os.path import join, dirname, realpath
import speech_service.whisper_model as whisper_model

app = FastAPI()
UPLOADS_PATH = join(dirname(realpath(__file__)), 'static\\uploads')
WHISPER = whisper_model.init_whisper("medium")

from speech_service.speech_main import *


@app.get("/")
async def root():
    return {"message": "Hello World"}
