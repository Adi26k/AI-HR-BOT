from app.models import database
import numpy as np
def get_all_jobpostings():
    container = database.get_container_client("JobPostings")
    items = list(container.read_all_items())
    return items

def get_optimal_job(application_id):
    print("APPLICATION ID",application_id)
    container = database.get_container_client("Applications")
    item = container.read_item(item=application_id, partition_key=application_id)
    application_embedding = item.get('vector_embedding')
    list_job_embeddings = get_all_jobpostings()
    highest_score = -1
    best_job = None
    for item in list_job_embeddings:
        job_embedding = item.get('vector_embedding')
        job_vector = np.array(job_embedding)
        resume_vector = np.array(application_embedding)
        similarity_score = float(np.dot(job_vector, resume_vector) / (np.linalg.norm(job_vector) * np.linalg.norm(resume_vector)))
        
        if similarity_score > highest_score:
            highest_score = similarity_score
            best_job = item
    highest_score = round(highest_score * 100, 2)
    return {'Job ID':best_job.get('id'),'Job Title':best_job.get('job_title'),'Match Score':highest_score}

async def get_ranked_applications(request):
    # Fetch all applications sorted by match_score (descending)
    query = "SELECT * FROM c ORDER BY c.scoring.match_score DESC"
    applications_container = database.get_container_client("Applications")
    jobpostings_container = database.get_container_client("JobPostings")
    applications = list(applications_container.query_items(query, enable_cross_partition_query=True))

    # Fetch all job postings and create a job_id -> job_title mapping
    job_query = "SELECT c.id, c.job_title FROM c"
    job_postings = list(jobpostings_container.query_items(job_query, enable_cross_partition_query=True))
    job_map = {job["id"]: job.get("job_title", "Unknown Job") for job in job_postings}

    # Prepare response data
    response_data = []
    for app in applications:
        response_data.append({
            "Application ID": app["id"],
            "Job ID": app["job_id"],
            "Job Title": job_map.get(app["job_id"], "Unknown Job"),
            "Match Score": app["scoring"]["match_score"],
            "Justification": app["scoring"]["justification"]
        })

    return {"rankings": response_data}