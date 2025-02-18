from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from azure.cosmos import CosmosClient, PartitionKey
import os
cosmos_client = CosmosClient(
url=os.getenv('COSMOS_DB_URL'),
credential=os.getenv('COSMOS_DB_KEY')
)
database_name = 'HrResumeParser'
database = cosmos_client.create_database_if_not_exists(database_name)
# -------------------------------
# JobPostings Collection Models
# -------------------------------

class ExtractedRequirements(BaseModel):
    skills: List[str]
    experience: List[str] = None
    education: List[str] = None
    additional_requirements: List[str]

class JobPostings(BaseModel):
    id: str = Field(alias="id")
    hr_id: str
    job_title: str
    job_description: str
    #jd_pdf_url: strS
    extracted_requirements: ExtractedRequirements
    vector_embedding: List[float]
    #pinecone_id: str
    #metadata: Dict[str, Any]

# -------------------------------
# Candidates Collection Model
# -------------------------------

class Candidate(BaseModel):
    id: str = Field(alias="_id")
    name: str
    email: str
    metadata: Dict[str, Any] = {}

# -------------------------------
# Applications Collection Models
# -------------------------------

class ExtractedData(BaseModel):
    skills: List[str]
    experience: Optional[str] = None
    education: Optional[str] = None
    soft_skills: List[str] 

class Scoring(BaseModel):
    match_score: float
    justification: str

class Application(BaseModel):
    id: str = Field(alias="id")
    candidate_id: str
    job_id: str
    extracted_data: ExtractedData
    vector_embedding: List[float]
    scoring: Scoring
