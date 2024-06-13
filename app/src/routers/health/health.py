import logging

from fastapi import APIRouter

api_logger = logging.getLogger('api_logger')

router = APIRouter()


@router.get("/health", tags=['health'], status_code=200)
def health() -> dict:
    api_logger.info('I\'m very healthy')
    return {
        'status': 200,
    }
