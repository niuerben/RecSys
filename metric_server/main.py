import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .evaluator import ML1MEvaluator, REFERENCE_ROWS, SubmittedFile
except ImportError:
    from evaluator import ML1MEvaluator, REFERENCE_ROWS, SubmittedFile


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ml-1m", "ml-1m")
STATIC_DIR = os.path.join(BASE_DIR, "static")


app = FastAPI(title="MovieLens-1M Metric Server")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_evaluator() -> ML1MEvaluator:
    return ML1MEvaluator(DATA_DIR)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/standard")
def standard():
    evaluator = get_evaluator()
    return {"standard": evaluator.standard_summary(), "table": REFERENCE_ROWS}


@app.post("/api/evaluate")
async def evaluate(
    method_name: str = Form("Your Result"),
    files: List[UploadFile] = File(...),
):
    submitted = []
    for file in files:
        submitted.append(SubmittedFile(name=file.filename or "upload.csv", content=await file.read()))

    try:
        return get_evaluator().evaluate_submission_files(submitted, method_name=method_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
