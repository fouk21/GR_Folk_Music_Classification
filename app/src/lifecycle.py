import logging
import os
import toml

from contextlib import asynccontextmanager
from fastapi import FastAPI

api_logger = logging.getLogger('api_logger')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Handle startup needs here
    api_logger.info("Handle startup needs")

    # Extract UPLOADS dir and create it
    CURRENT_DIR = os.getenv('INDEX_PATH')

    config_file_path = f'{CURRENT_DIR}/../config.toml'
    config = toml.load(config_file_path)

    upload_dir = f'{CURRENT_DIR}/../{config["settings"]["uploads_dir"]}'
    os.makedirs(upload_dir, exist_ok=True)
    os.environ['UPLOADS_DIR'] = str(upload_dir)

    api_logger.info('Application startup: Set the UPLOADS directory')

    yield

    # Handle shutdown needs here
    api_logger.info("Handle shutdown needs")
