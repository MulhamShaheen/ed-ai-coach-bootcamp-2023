import os
import main
from .whisper_model import speech_analysis
from main import app as main_app
from fastapi import FastAPI, File, UploadFile
import aiofiles

app = main_app


@app.post("/speech")
async def create_upload_file(file: UploadFile):

    async with aiofiles.open(os.path.join(main.UPLOADS_PATH, file.filename), 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write

        output = speech_analysis(os.path.join(main.UPLOADS_PATH, file.filename))

    return {"filename": file.filename, "output": output}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hellssso {name}"}
