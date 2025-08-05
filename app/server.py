from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from langserve import add_routes
from app.rag_chain import final_chain
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langserve import add_routes
from app.rag_chain import final_chain
import os
import shutil
import subprocess

app = FastAPI()

@app.post("/load-and-process-pdfs")
async def load_and_process_pdfs():
    try:
        subprocess.run(["python", "./rag-data-loader/rag_load_and_process.py"], check=True)
        return {"message": "PDFs loaded and processed successfully"}
    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to execute script: {e}"}

pdf_directory = "./pdf-documents"
os.makedirs(pdf_directory, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.mount("/rag/static", StaticFiles(directory="./pdf-documents"), name="static")
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    for file in files:
        try:
            file_path = os.path.join(pdf_directory, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    
    return {"message": "Files uploaded successfully", "filenames": [file.filename for file in files]}


# Edit this to add the chain you want to add
add_routes(app, final_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



