import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI

api_logger = logging.getLogger('api_logger')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Handle startup needs here
    api_logger.info("Handle startup needs")

    yield

    # Handle shutdown needs here
    api_logger.info("Handle shutdown needs")
