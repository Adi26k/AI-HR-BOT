from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from fastapi.params import Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.job_parser import process_job_posting, extract_text_from_pdf
from app.resume_parser import process_application
from app.utils import get_all_jobpostings, get_optimal_job, get_ranked_applications

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

@router.get("/find_best_job")
async def find_best_job(application_id: str = Query(...)):
    best_jobs = get_optimal_job(application_id)
    return JSONResponse(content=best_jobs)

@router.get("/rankings")
async def get_ranking_applications(request: Request):
    ranked_applications = await get_ranked_applications(request)
    rankings = ranked_applications.get("rankings", [])
    return templates.TemplateResponse("ranking.html", {"request": request, "rankings": rankings})