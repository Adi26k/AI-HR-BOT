from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os
from openai import AzureOpenAI
from app.models import Application, Scoring, JobPostings, ExtractedRequirements, database, ExtractedData
import json
import numpy as np
from app.tools import parse_jd, parse_application
from autogen_agentchat.conditions import TextMentionTermination,MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from app.utils import get_all_jobpostings, get_optimal_job, get_ranked_applications

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("OPENAI_RESOURCE_ENDPOINT")
)
from typing import Sequence, Union
from autogen_agentchat.messages import AgentEvent, ChatMessage
import re
from pydantic import BaseModel

class RoutingDecision(BaseModel):
    agent: str

def selector_func(messages: Sequence[Union[AgentEvent, ChatMessage]]) -> str | None:
    
    if not messages:
        return "PlannerAgent"  # Start with PlannerAgent

    last_message = messages[-1]
    source = last_message.source.lower()
    content = last_message.content.strip()
    
    # Debug print to monitor the content
    print(f"🟢 Last message from {source}: '{content}'")
    
    # When PlannerAgent speaks, try to extract the next agent from structured output.
    if source == "planneragent":
        try:
            decision = RoutingDecision.model_validate_json(content)
            return decision.agent
        except Exception as e:
            # Fallback: try extracting using regex if not valid JSON.
            match = re.search(r"nextagent:\s*([a-zA-Z0-9]+)", content, re.IGNORECASE)
            if match:
                return match.group(1)
        return None  # Fallback to model selection if PlannerAgent's output is unclear.
    
    # When SupportAgent is active:
    if source == "supportagent":
        # If candidate explicitly instructs to apply with a job id, route to ResumeParser.
        if "apply" in content and re.search(r"\b\d+\b", content):
            return "ResumeParser"
        # Otherwise, let SupportAgent continue handling the request.
        return "TERMINATE"

    # After ResumeParser processes an application, pass control back to SupportAgent.

    
    # If JobParser (HR job posting) is active, move to SaveJobPosting.
    if source == "jobparser" or source == "resumeparser":
        return "SaveJobPosting"
    
    # If SaveJobPosting indicates termination, end the conversation.
    if source == "savejobposting" or "terminate" in content.lower():
        return "TERMINATE"
    
    # Default fallback.
    return None



SupportAgent = AssistantAgent(
    name="SupportAgent",
    model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_deployment="gpt-4o",
        azure_endpoint=os.getenv("OPENAI_RESOURCE_ENDPOINT"),
    ),
    description="Assists candidates in finding jobs and HR in ranking applications.",

    system_message="""You assist job seekers and HR professionals.
    
    - If the user is a **Candidate**, assist them in:
      - Listing jobs (`list_jobs`).
      - Finding the best-matching job (`get_optimal_job`).
      - Applying to a job (forward to `ResumeParser`).
    
    - If the user is **HR**, assist in:
      - Ranking applications (`get_ranked_applications`).
    
    - Validate parameters before calling tools.
    - Only return tool outputs as responses.
    """,

    tools=[
        FunctionTool(
            name="list_jobs",
            func=get_all_jobpostings,
            description="Lists all available job postings."
        ),
        FunctionTool(
            name="get_optimal_job",
            func=get_optimal_job,  # Now properly typed
            description="Finds the best job for a candidate. Requires `application_id` (string)."
        ),
        FunctionTool(
            name="get_ranked_applications",
            func=get_ranked_applications,  # Now properly typed
            description="Ranks job applications for HR."
        )
    ]
)





applicationAgent = AssistantAgent(
    name="ResumeParser",
    model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv('OPENAI_API_KEY'),
        api_version=os.getenv('OPENAI_API_VERSION'),
        azure_deployment='gpt-4o',
        azure_endpoint=os.getenv('OPENAI_RESOURCE_ENDPOINT'),
        response_format=Application
    ),
    description="Processes candidate resumes and generates structured job applications for a given job posting.",
    system_message="""You are an HR assistant specialized in processing resumes and generating job applications.
    
    **Responsibilities:**
    - Extracts relevant details from resumes.
    - Matches candidate information with job requirements.
    - Generates structured job applications linked to a specific job_id.

    **Important Instructions:**
    - Always require a job_id before processing an application.
    - Ensure extracted data is structured for database storage.
    - Pass the application data to the SaveJobPostingTool upon completion as is without any additional text
    """,
)


JobPostingsAgent = AssistantAgent(
    name="JobParser",
    model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv('OPENAI_API_KEY'),
        api_version=os.getenv('OPENAI_API_VERSION'),
        azure_deployment='gpt-4o',
        azure_endpoint=os.getenv('OPENAI_RESOURCE_ENDPOINT'),
        response_format=JobPostings
    ),
    description="Parses job descriptions and extracts structured job posting details.",
    system_message="""You are an HR assistant specializing in job postings.

    **Responsibilities:**
    - Analyze job descriptions.
    - Extract key details like job title, skills, requirements, and company information.
    - Store the structured job posting for future applications.

    **Important Instructions:**
    - Ensure extracted data is structured and complete.
    - Pass the job posting data to the SaveJobPostingTool upon completion.
    """,
)



SaveJobPostingTool = AssistantAgent(
    name="SaveJobPosting",
    model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv('OPENAI_API_KEY'),
        api_version=os.getenv('OPENAI_API_VERSION'),
        azure_deployment='gpt-4o',
        azure_endpoint=os.getenv('OPENAI_RESOURCE_ENDPOINT'),
    ),
    description="Saves job postings and applications to the database using the appropriate tool.",
    system_message="""You are responsible for storing job postings and applications in the database.

    **Responsibilities:**
    - Identify whether the incoming data is a job posting or an application.
    - Use the appropriate tool to store the data.
    - Confirm successful storage and return a termination signal.

    **Important Instructions:**
    - If the data contains a `job_id`, it represents a job application. Use `parse_application`.
    - Otherwise, it is a job posting. Use `parse_jd`.
    - Output of the Tool should be output of agent as is without any adiitional Text.
    IMPORTANT: You will only get data from ResumeParser and JobParser as is.
    """,
    tools=[
        FunctionTool(name="parse_jd", func=parse_jd, 
                     description="Parses and stores job descriptions in the database."),
        FunctionTool(name="parse_application", func=parse_application,
                     description="Parses and stores job applications in the database."),
    ],
)

PlannerAgent = AssistantAgent(
    name="PlannerAgent",
    model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_deployment="gpt-4o",
        azure_endpoint=os.getenv("OPENAI_RESOURCE_ENDPOINT"),
        response_format=RoutingDecision
    ),
    description="Routes user requests to the correct agent.",
    
    system_message="""You are responsible for directing requests to the correct agent.
    
    - If the user is **HR** and wants to create a job post → assign to `JobParser`.
    - If the user is a **Candidate** and mentions a job ID (e.g., job_1234) → forward to `ResumeParser`.
    - If the user is a **Candidate** and is looking for jobs or wants to know best job → forward to `SupportAgent`.
    - Prevent redundant agent loops.
    - Agents should **only respond if relevant to their role**.
    """,
    tools=[]
)


text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination

team = SelectorGroupChat(
    [PlannerAgent, SupportAgent, applicationAgent, JobPostingsAgent, SaveJobPostingTool],
    model_client=AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_deployment="gpt-4o",
        azure_endpoint=os.getenv("OPENAI_RESOURCE_ENDPOINT"),
    ),
    termination_condition=TextMentionTermination("TERMINATE"),

    selector_prompt="""You are a speaker selector for an HR automation system.

- The `PlannerAgent` ALWAYS decides which agent should handle the request.
- If the user is HR, their request MUST be routed to `JobParser`.
- If the user is a Candidate:
  - Direct them to `SupportAgent` for job listings.
  - If they select a job, send them to `ApplicationAgent` for applying.
- `SaveJobPostingTool` should only be used when a job needs to be stored.

DO NOT allow agents to interfere once `PlannerAgent` has assigned a task.
- Ensure the output of one agent is passed to another agent as is without any modifications.
""",
selector_func=selector_func
)
