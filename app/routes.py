from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Form
from fastapi.params import Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.job_parser import process_job_posting, extract_text_from_pdf
from app.resume_parser import process_application
from app.utils import get_all_jobpostings, get_optimal_job, get_ranked_applications
from autogen_agentchat.ui import Console
from app.agents import team
from pydantic import BaseModel
import json
import ast
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


class ChatRequest(BaseModel):
    user_type: str  # "HR" or "Candidate"
    user_input: str

@router.post("/chat")
async def chat_with_agents(request: ChatRequest):
    """
    The PlannerAgent determines the correct agent for the task.
    """
    result  = await Console(team.run_stream(task=f"Role: {request.user_type}\nMessage: {request.user_input}"))
    message = result.messages[-1].content
    try:
        response = ast.literal_eval(message)  # Convert to dictionary
    except (ValueError, SyntaxError):
        response = json.loads(message)  # Try JSON parsing

    return JSONResponse(content=response)


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
    ranked_applications = await get_ranked_applications()
    rankings = ranked_applications.get("rankings", [])
    return templates.TemplateResponse("ranking.html", {"request": request, "rankings": rankings})