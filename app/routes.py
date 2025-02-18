from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.job_parser import process_job_posting, extract_text_from_pdf
from app.resume_parser import process_application
from app.utils import get_all_jobpostings

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

    response = await process_job_posting(file_path)
    return JSONResponse(content=response)

@router.get("/job_postings", response_class=HTMLResponse)
async def job_postings(request: Request):
    job_postings = get_all_jobpostings()
    return templates.TemplateResponse("job_postings.html", {"request": request, "job_postings": job_postings})

@router.post("/upload_application")
async def upload_application(resume: UploadFile = File(...), jobid: str = Form(...)):
    if resume.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    file_path = f"/tmp/{resume.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await resume.read())

    response = await process_application(file_path, jobid)
    return JSONResponse(content=response)