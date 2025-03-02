from app.models import database
import numpy as np
import json
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from app.models import Scoring

ScoringAgent = AssistantAgent(
    name="RankingAgent",
    model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv('OPENAI_API_KEY'),
        api_version=os.getenv('OPENAI_API_VERSIOM'),
        azure_deployment='gpt-4o',
        azure_endpoint=os.getenv('OPENAI_RESOURCE_ENDPOINT'),
        response_format=Scoring
    ),
    system_message="You are an agent whose job is to provide match score and justification when two embeddings are compared.",
)

def get_all_jobpostings():
    container = database.get_container_client("JobPostings")
    items = list(container.read_all_items())
    cleaned_items = [{k: v for k, v in item.items() if k != 'vector_embedding'} for item in items]
    return cleaned_items


def get_optimal_job(application_id: str) -> dict:
    application = application_id.strip()
    print("APPLICATION ID:", application)

    # Fetch application embedding
    container = database.get_container_client("Applications")
    item = container.read_item(item=application, partition_key=application)

    application_embedding = item.get("vector_embedding")
    if application_embedding is None:
        print(f"⚠️ ERROR: Application {application_id} has no vector embedding!")
        return {"error": f"Application {application_id} has no vector embedding."}

    application_vector = np.array(application_embedding)

    jd_container = database.get_container_client("JobPostings")
    jd_items = list(jd_container.read_all_items())
    list_job_embeddings = jd_items
    if not list_job_embeddings:
        print("⚠️ ERROR: No job postings found with embeddings!")
        return {"error": "No job postings found with embeddings."}

    highest_score = -1
    best_job = None

    for job in list_job_embeddings:
        job_id = job.get("id")
        job_embedding = job.get("vector_embedding")

        print(f"ITEM jobpost_{job_id}")

        # Ensure job has a valid embedding
        if job_embedding is None:
            print(f"⚠️ WARNING: Job {job_id} has no vector embedding. Skipping.")
            continue

        job_vector = np.array(job_embedding)

        # Compute similarity safely
        norm_product = np.linalg.norm(job_vector) * np.linalg.norm(application_vector)
        if norm_product == 0:  # Avoid division by zero
            print(f"⚠️ WARNING: Zero norm encountered for job {job_id}. Skipping.")
            continue

        similarity_score = float(np.dot(job_vector, application_vector) / norm_product)
        print(f"✅ SIMILARITY SCORE for {job_id}: {similarity_score}")

        if similarity_score > highest_score:
            highest_score = similarity_score
            best_job = job

    # Handle case where no valid job is found
    if best_job is None:
        return {"error": "No matching job found for this application."}

    highest_score = round(highest_score * 100, 2)

    return {
        "Job ID": best_job.get("id"),
        "Job Title": best_job.get("job_title", "Unknown"),
        "Match Score": highest_score,
    }



async def get_ranked_applications() -> dict:
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



async def scoring_agent_func(job_id,resume_embedding,extracted_resume_data):
    container = database.get_container_client("JobPostings")
    jd = container.read_item(item=job_id, partition_key=job_id)
    job_embedding = jd.get('vector_embedding')
    extracted_job_data = jd.get('extracted_requirements')
    job_vector = np.array(job_embedding)
    resume_vector = np.array(resume_embedding)
    similarity_score = float(np.dot(job_vector, resume_vector) / (np.linalg.norm(job_vector) * np.linalg.norm(resume_vector)))
    
    # Use tool to retrieve similar candidates for context (for justification)
    
    print("SIMILARITY SCORE",similarity_score)
    # Construct prompt with extracted data and similar candidate info
    prompt = f"""
    Given the extracted job requirements:
    {json.dumps(extracted_job_data, indent=2)}
    
    And given the extracted resume details:
    {json.dumps(extracted_resume_data, indent=2)}
    
    Generate a match score (0-100) and provide a detailed justification explaining Whether the candidate is  suitable for job.
    """
    
    response = await ScoringAgent.on_messages(
        [TextMessage(content=prompt, source="user")],
        cancellation_token=CancellationToken()
    )
    response_content = response.chat_message.content
    response_dict = json.loads(response_content)
    # Incorporate the advanced similarity score into the final match score.
    response_dict["match_score"] = (round(similarity_score * 100, 2) + response_dict.get("match_score", 0.0))/ 2
    return response_dict
