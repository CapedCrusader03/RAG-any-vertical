import os
import uuid
import json
import logging
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import the ingestion functions and the progress registry
from scripts.ingest import ingest_document, ingestion_tasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Regulated RAG Document Ingestion Portal")

# Ensure target directories exist
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_corpus")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper to write files in chunks
def save_upload_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        while True:
            chunk = upload_file.file.read(1024 * 1024) # 1MB chunks
            if not chunk:
                break
            buffer.write(chunk)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend template index.html not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    source_url: str = Form(""),
    roles: str = Form(...),          # Expecting comma-separated roles
    jurisdictions: str = Form(...)    # Expecting comma-separated jurisdictions
):
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files are supported.")

    # Parse metadata lists
    allowed_roles = [r.strip() for r in roles.split(",") if r.strip()]
    allowed_jurisdictions = [j.strip() for j in jurisdictions.split(",") if j.strip()]

    if not allowed_roles or not allowed_jurisdictions:
        raise HTTPException(status_code=400, detail="Roles and jurisdictions must contain at least one valid value.")

    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    ingestion_tasks[task_id] = {
        "state": "pending",
        "log": "Initializing upload task...",
        "percentage": 0
    }

    # Define paths
    safe_filename = f"{task_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    meta_path = os.path.join(UPLOAD_DIR, f"{task_id}_{os.path.splitext(file.filename)[0]}.meta.json")

    try:
        # Save file to disk
        save_upload_file(file, file_path)

        # Write corresponding meta json
        meta_content = {
            "title": title,
            "source_url": source_url,
            "allowed_roles": allowed_roles,
            "allowed_jurisdictions": allowed_jurisdictions
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=2)

        # Schedule ingestion in the background
        background_tasks.add_task(ingest_document, file_path, meta_path, task_id)
        
        logger.info(f"Scheduled task {task_id} for file {file.filename}")
        return JSONResponse(content={"task_id": task_id, "filename": file.filename})
    except Exception as e:
        logger.error(f"Failed to initialize upload for {file.filename}: {e}")
        ingestion_tasks[task_id] = {
            "state": "failed",
            "log": f"Failed to initialize upload: {e}",
            "percentage": 100
        }
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    task = ingestion_tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return JSONResponse(content=task)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting local RAG ingestion web server...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
