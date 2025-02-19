from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os
from openai import AzureOpenAI
from app.models import Application, Scoring, JobPostings, ExtractedRequirements, database, ExtractedData
import json
import numpy as np

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("OPENAI_RESOURCE_ENDPOINT")
)

applicationAgent = AssistantAgent(
    name="ResumeParser",
    model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv('OPENAI_API_KEY'),
        api_version=os.getenv('OPENAI_API_VERSIOM'),
        azure_deployment='gpt-4o',
        azure_endpoint=os.getenv('OPENAI_RESOURCE_ENDPOINT'),
        response_format=Application
    ),
    system_message="You are an expert HR agent which analyzes resume document.",
)

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



JobPostingsAgent = AssistantAgent(
    name="JobParser",
    model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv('OPENAI_API_KEY'),
        api_version=os.getenv('OPENAI_API_VERSIOM'),
        azure_deployment='gpt-4o',
        azure_endpoint=os.getenv('OPENAI_RESOURCE_ENDPOINT'),
        response_format=JobPostings
    ),
    system_message="You are an expert HR agent which analyzes job description document.",
)

async def application_agent(text,metadata):
    response = await applicationAgent.on_messages(
        [TextMessage(content=f"""
            Extract key details from resume in given format. 
            Generate unique id for the candiate id wherever required.
             The format of job id shoud be in the format of "app_id_1
            {text}
        """, source="user")],
        cancellation_token=CancellationToken()
    )
    response_content = response.chat_message.content
    response_dict = json.loads(response_content)
    application_embedding = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    response_dict['job_id'] = str(metadata.get('job_id'))
    response_dict["scoring"] = await scoring_agent_func(job_embedding=metadata.get("job_embedding"), 
                                                  resume_embedding=application_embedding.model_dump()['data'][0]['embedding'], 
                                                  extracted_job_data=metadata.get("extracted_job_data"), 
                                                  extracted_resume_data=response_dict.get("extracted_data"))
    return response_dict, application_embedding

async def job_posting_agent(text):
    response = await JobPostingsAgent.on_messages(
        [TextMessage(content=f"""
            Generate job posting from job description: based on given format. 
            Generate unique id for the job posting wherever required.
            The format of job id shoud be in the format of "job_id_1"
            {text}
        """, source="user")],
        cancellation_token=CancellationToken()
    )
    response_content = response.chat_message.content
    response_dict = json.loads(response_content)
    job_posting_embedding = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )

    return response_dict, job_posting_embedding

async def scoring_agent_func(job_embedding, resume_embedding, extracted_job_data, extracted_resume_data):
    # Calculate cosine similarity
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


