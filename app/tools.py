
from openai import AzureOpenAI
from app.models import JobPostings, ExtractedRequirements, database,Application,ExtractedData,Scoring
from azure.cosmos import PartitionKey
from io import BytesIO
import json
import os
from app.utils import scoring_agent_func
import pprint
client = AzureOpenAI(
  api_key = os.getenv("OPENAI_API_KEY"),  
  api_version = "2024-06-01",
  azure_endpoint =os.getenv("OPENAI_RESOURCE_ENDPOINT") 
)

def parse_jd(data_text:str) -> dict:
    print("DATA TEXT",data_text)
    response_dict = json.loads(data_text)
    job_posting_embedding = client.embeddings.create(
            input=data_text,
            model="text-embedding-3-large"
        )
        # Create JobPostings instance
    job_posting = JobPostings(
        id=response_dict.get("id"),
        hr_id=response_dict.get("hr_id"),
        job_title=response_dict.get("job_title"),
        job_description=response_dict.get("job_description"),
        jd_pdf_url=response_dict.get("jd_pdf_url"),
        extracted_requirements=ExtractedRequirements(
            skills=response_dict.get("extracted_requirements", {}).get("skills", []),
            experience=response_dict.get("extracted_requirements", {}).get("experience", None),
            education=response_dict.get("extracted_requirements", {}).get("education", None),
            additional_requirements=response_dict.get("extracted_requirements", {}).get("additional_requirements", [])
        ),
        vector_embedding=job_posting_embedding.model_dump()['data'][0]['embedding'],
        pinecone_id=response_dict.get("pinecone_id"),
        upload_date=response_dict.get("upload_date", None),
        metadata=response_dict.get("metadata", {})
    )
    vector_embedding_policy = {
        "vectorEmbeddings": [ 
    { 
        "path":"/" + 'vector_embedding',
        "dataType":"float32",
        "distanceFunction":"cosine",
        "dimensions":len(job_posting.vector_embedding),
    }, 
        ]
    }

    container = database.create_container_if_not_exists(
        id='JobPostings',
        partition_key=PartitionKey(path="/id"),
        vector_embedding_policy=vector_embedding_policy,  # Choose a suitable partition key
        offer_throughput=400  # Adjust throughput as needed
        )
    jd=container.upsert_item(job_posting.model_dump(by_alias=True))
    print("JD",jd)
    # Store the response dictionary as extracted_requirements
    return {
        "job_title": jd.get("job_title"),
        "job_description": jd.get("job_description"),
        }



async def parse_application(data_text:str) -> dict:
    response_dict = json.loads(data_text)
    pprint.pprint(response_dict)
    print("RESPONSE DICT JOB ID",response_dict['job_id'])
    application_embedding = client.embeddings.create(
            input=data_text,
            model="text-embedding-3-large"
        )
    container = database.get_container_client("JobPostings")
    item = container.read_item(item=response_dict['job_id'], partition_key=response_dict['job_id'])
    response_dict['scoring'] = await scoring_agent_func(response_dict['job_id'],
                                                        application_embedding.model_dump()['data'][0]['embedding'],
                                                        response_dict['extracted_data'])
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
    print("lenght",len(appllication.vector_embedding))
    container = database.create_container_if_not_exists(
        id='Applications',
        vector_embedding_policy=vector_embedding_policy,
        partition_key=PartitionKey(path="/id"),  # Choose a suitable partition key
        offer_throughput=400  # Adjust throughput as needed
        )
    application_details=container.upsert_item(appllication.model_dump(by_alias=True))
    # Store the response dictionary as extracted_requirements
    return {
        "id": application_details.get("id"),
        "match_score": application_details.get("scoring").get('match_score'),
        "justification": application_details.get("scoring").get('justification'),
    }






    
    