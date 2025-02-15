from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY') or 'your_default_secret_key'
    DEBUG = os.getenv('DEBUG', 'False').lower() in ['true', '1']
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL') or 'sqlite:///site.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_RESOURCE_ENDPOINT = os.getenv('OPENAI_RESOURCE_ENDPOINT')
    OPENAI_API_VERSION= os.getenv('OPENAI_API_VERSION')