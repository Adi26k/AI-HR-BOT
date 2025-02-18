from app.models import database
from app.ranker import JobMatching

def get_all_jobpostings():
    container = database.get_container_client("JobPostings")
    items = list(container.read_all_items())
    return items

def get_optimal_job(application_embedding):
    matcher = JobMatching(get_all_jobpostings(),use_hnsw=True)
    best_jobs = matcher.find_best_jobs(application_embedding)
    return best_jobs