from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
   
app = FastAPI()


@app.post("/bot")
async def ab(sent:str):
    return sent[0:1]
