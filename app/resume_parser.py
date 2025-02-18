

from app.models import Application,Scoring,database,ExtractedData
from azure.cosmos import  PartitionKey
from app.agents import application_agent
from app.job_parser import extract_text_from_pdf

async def process_application(file,jobid):
    print("JOB ID",jobid)
    file_uploaded = True
    if file_uploaded:
        with open(file, 'rb') as file:
            file_bytes = file.read()
        text = extract_text_from_pdf(file_bytes)
        container = database.get_container_client("JobPostings")
        job_posting = container.read_item(item=jobid, partition_key=jobid)
        metadata = {'job_embedding':job_posting.get('vector_embedding'),'extracted_job_data':job_posting.get('extracted_requirements'),'job_id':jobid}
        response_dict, application_embedding = await application_agent(text,metadata)
        # Create JobPostings instance
        appllication = Application(
       id=response_dict.get("id"),
      candidate_id=response_dict.get("candidate_id"),  # Assuming candidate_id is available in response_dict
      job_id=response_dict.get("job_id"),
      extracted_data=ExtractedData(
        skills=response_dict.get("extracted_data", {}).get("skills", []),
        experience=response_dict.get("extracted_data", {}).get("experience", None),
       education=response_dict.get("extracted_data", {}).get("education", None),
        soft_skills=response_dict.get("extracted_data", {}).get("soft_skills", [])
    ),
    vector_embedding=application_embedding.model_dump()['data'][0]['embedding'],
     scoring=Scoring(
        match_score=response_dict.get("scoring", {}).get("match_score", 0.0),  # Default to 0.0 if not found
        justification=response_dict.get("scoring", {}).get("justification", "")
    ), # Set the current date and time
)
        vector_embedding_policy = {
          "vectorEmbeddings": [ 
        { 
            "path":"/" + 'vector_embedding',
            "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":len(appllication.vector_embedding),
        }, 
            ]
        }
        container = database.create_container_if_not_exists(
          id='Applications',
          vector_embedding_policy=vector_embedding_policy,
          partition_key=PartitionKey(path="/id"),  # Choose a suitable partition key
          offer_throughput=400  # Adjust throughput as needed
          )
        application_details=container.upsert_item(appllication.model_dump(by_alias=True))
        # Store the response dictionary as extracted_requirements
        return {
            "match_score": application_details.get("scoring").get('match_score'),
            "justification": application_details.get("scoring").get('justification'),
        }
