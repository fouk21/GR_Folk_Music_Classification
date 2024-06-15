import logging
import os

from fastapi import (
    APIRouter,
    File,
    UploadFile,
)
from fastapi.responses import JSONResponse

api_logger = logging.getLogger('api_logger')

router = APIRouter()


@router.post("/uploads", tags=['uploads'])
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file
    UPLOADS_DIR = os.getenv('UPLOADS_DIR')
    file_path = os.path.join(UPLOADS_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return JSONResponse(
        content={
            'filename': file.filename,
            'message': 'File uploaded successfully',
        }
    )
