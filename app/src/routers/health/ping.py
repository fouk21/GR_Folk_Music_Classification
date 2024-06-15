import logging

from fastapi import APIRouter

api_logger = logging.getLogger('api_logger')

router = APIRouter()


@router.get("/ping", tags=['dataset'])
def ping() -> str:
    return 'pong'
