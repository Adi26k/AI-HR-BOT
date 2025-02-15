from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from io import BytesIO
import pdfplumber
import asyncio
from openai import AzureOpenAI
from app.models import JobPostings, ExtractedRequirements,database
from azure.cosmos import CosmosClient, PartitionKey

import json
import os
model_client = AzureOpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=os.getenv('OPENAI_API_KEY'),
    api_version=os.getenv('OPENAI_API_VERSIOM'),
    azure_deployment='gpt-4o',
    azure_endpoint=os.getenv('OPENAI_RESOURCE_ENDPOINT'),
    response_format=JobPostings
)
client = AzureOpenAI(
  api_key = os.getenv("OPENAI_API_KEY"),  
  api_version = "2024-06-01",
  azure_endpoint =os.getenv("OPENAI_RESOURCE_ENDPOINT") 
)

async def process_resume(file):
    file_uploaded = True
    if file_uploaded:
        with open(file, 'rb') as file:
            file_bytes = file.read()
        text = extract_text_from_pdf(file_bytes)
        agent = AssistantAgent(
            name="JobParser",
            model_client=model_client,
            system_message="You are an expert HR agent which analyzes job description document.",
        )
        response = await agent.on_messages(
            [TextMessage(content=f"""
                        Generate job posting from job description: based on given format. 
                         Generate unique id for the job posting wherever required.
                         {text}
                        """, source="user")],
            cancellation_token=CancellationToken()
        )   
        response_content = response.chat_message.content
        response_dict = json.loads(response_content)
        jd_embedding = client.embeddings.create(
        input = text,
        model= "text-embedding-3-large"
        )
        
        # Create JobPosting instance
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
            vector_embedding=jd_embedding.model_dump()['data'][0]['embedding'],
            metadata=response_dict.get("metadata", {})
        )
        vector_embedding_policy = {
          "vectorEmbeddings": [ 
        { 
            "path":"/" + 'vector_embedding',
            "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":response_dict.get("vector_embedding", []),
        }, 
            ]
        }

        container = database.create_container_if_not_exists(
          id='JobPostings',
          partition_key=PartitionKey(path="/jd"),
            vector_embedding_policy=vector_embedding_policy,  # Choose a suitable partition key
          offer_throughput=400  # Adjust throughput as needed
          )
        jd=container.upsert_item(job_posting.model_dump(by_alias=True))
        # Store the response dictionary as extracted_requirements
        return {
            "job_title": jd.get("job_title"),
            "job_description": jd.get("job_description"),
        }


def extract_text_from_pdf(file_bytes):
    text = ''
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def convert_pdf_to_bytes(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

