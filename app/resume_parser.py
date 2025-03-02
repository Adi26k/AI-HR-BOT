

from app.models import Application,Scoring,database,ExtractedData
from azure.cosmos import  PartitionKey
from app.agents import applicationAgent
import json
import os
import numpy as np
from openai import AzureOpenAI
from app.job_parser import extract_text_from_pdf
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("OPENAI_RESOURCE_ENDPOINT")
)


async def process_application(file,jobid):
    print("JOB ID",jobid)
    file_uploaded = True
    if file_uploaded:
        with open(file, 'rb') as file:
            file_bytes = file.read()
        text = extract_text_from_pdf(file_bytes)
        container = database.get_container_client("JobPostings")
        job_posting = container.read_item(item=jobid, partition_key=jobid)
        response = await applicationAgent.on_messages(
        [TextMessage(content=f"""
            Extract key details from resume in given format. 
            Generate unique id for the candiate id wherever required.
             The job_id should be {jobid}
            {text}
        """, source="user")],
        cancellation_token=CancellationToken()
    )
        response_content = response.chat_message.content
        response_dict = json.loads(response_content)
        return response_dict
    

