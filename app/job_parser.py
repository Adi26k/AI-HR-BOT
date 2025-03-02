
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from openai import AzureOpenAI
from app.models import JobPostings, ExtractedRequirements, database
from azure.cosmos import PartitionKey
from app.agents import JobPostingsAgent
import pdfplumber
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
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
        response = await JobPostingsAgent.on_messages(
        [TextMessage(content=f"""
            Generate job posting from job description: based on given format. 
            Generate unique id for the job posting wherever required.
            The format of job id shoud be in the format of "job_id_1"
            {text}
        """, source="user")],
        cancellation_token=CancellationToken()
    )
        print("RESPONSE",response)
        response_content = response.chat_message.content
        print("RESPONSE Content",response_content)
        response_dict = json.loads(response_content)
        return response_dict
        

def convert_pdf_to_bytes(file_path):
    with open(file_path, 'rb') as file:
        return file.read()
    
def extract_text_from_pdf(file_bytes):
    text = ''
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text



