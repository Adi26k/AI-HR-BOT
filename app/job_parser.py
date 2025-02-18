
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from openai import AzureOpenAI
from app.models import JobPostings, ExtractedRequirements, database
from azure.cosmos import PartitionKey
from app.agents import job_posting_agent
import pdfplumber
from io import BytesIO
import json
import os

client = AzureOpenAI(
  api_key = os.getenv("OPENAI_API_KEY"),  
  api_version = "2024-06-01",
  azure_endpoint =os.getenv("OPENAI_RESOURCE_ENDPOINT") 
)

async def process_job_posting(file):
    file_uploaded = True
    if file_uploaded:
        with open(file, 'rb') as file:
            file_bytes = file.read()
        text = extract_text_from_pdf(file_bytes)
        
        response_dict, job_posting_embedding = await job_posting_agent(text)
        #print("RESPONSE DICT",response_dict)
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
       

def convert_pdf_to_bytes(file_path):
    with open(file_path, 'rb') as file:
        return file.read()
    
def extract_text_from_pdf(file_bytes):
    text = ''
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text



