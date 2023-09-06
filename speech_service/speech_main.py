from fastapi import FastAPI, File, UploadFile
import aiofiles

from main import app as main_app

app = main_app


@app.post("/speech")
async def create_upload_file(file: UploadFile):

    async with aiofiles.open(os.path.join("../static/uploads"), 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write

    return {"filename": file.filename}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hellssso {name}"}
