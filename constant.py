import os
from dotenv import load_dotenv

load_dotenv()

ERROR_500 = "Oops! Looks like connection issue. Please try again"

PRATICE_ADD = "Practice add"
PRATICE_EDIT = "Practice edit"
PRATICE_LIST = "Practice list"
listing_screen = 'Listing Screen'
view_permission = 'View'
AUTH_ERROR = "You are not authorised to perform this action."
REPROCESSING_MESSAGE = "Reprocessing has been started."
role_not_found = "Role not found"

STATUS_MESSAGE = "Status updated successfully"
DOCUEMNT_NOT_FOUND = "Document not found"
STATUS_INREVIEW = "Please change status of document as 'IN REVIEW'"
INPUT_TEXT = "Input text is not correct"
FILE_NOT_FOUND = "file not found"
PAGE_NOT_FOUND = "page not found"
OWN_ACCOUNT_DELETE = "You cannot delete your own account."
REPORT_INPROGRESS = "REPORT IN PROGRESS"
REPORT_GENERATED = 'REPORT GENERATED'
EDIT_CODES = 'Edit SNOMED codes'
EDIT_REPORT = 'Edit report'
NO_CATEGORY = 'No Category'
REGISTER_EMAIL = "Register email already exists"
PERMISSION_MESSAGE = "You don't have permission to access this user details"
DETAILS_NOT_PROVIDED = "Details not valid"
SERVER_ERROR = "server error"
token_descritopn,autherization = 'Authentication token','Authorization'
done,new,in_progress,ai_processed,in_review,failed,document_revision = 'DONE','NEW','IN PROGRESS','AI PROCESSED','IN REVIEW','FAILED',"DOCUMENT REVISION"
bad_request,sucess = 'Bad Request','Success'
MEDICATION_DETAILS_CREATE,MEDICATION_DETAILS_UPDATE = 'MEDICATION_DETAILS_CREATE','MEDICATION_DETAILS_UPDATE'
approved = 'APPROVED'
category_action,reading_action,delete_reading = "CATEGORY_UPDATE","READING_UPDATE",'DELETE_READ'
region_name='eu-west-2'

aws_access_key_id=os.getenv('AWS_S3_ACCESS_KEY')
aws_secret_access_key=os.getenv('AWS_S3_SECRET_KEY')

application_json = "application/json"
AWS_BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
AWS_BEDROCK_MODEL_LLM="meta.llama3-8b-instruct-v1:0"
embedding_model_id = "amazon.titan-embed-text-v1"