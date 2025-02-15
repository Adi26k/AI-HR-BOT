from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.job_parser import process_resume, extract_text_from_pdf

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@router.post("/upload")
async def upload_file(resume: UploadFile = File(...)):
    if resume.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    file_path = f"/tmp/{resume.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await resume.read())

    response = await process_resume(file_path)
    return JSONResponse(content=response)