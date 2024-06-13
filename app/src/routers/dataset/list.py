import logging

from fastapi import APIRouter

api_logger = logging.getLogger('api_logger')

router = APIRouter()


@router.get("/dataset", tags=['dataset'])
def dataset_list() -> list:
    return [{
        "song": "sirtaki"
    }, {
        "song": "pontiako"
    }]


@router.get("/dataset/{id}", tags=['dataset'])
def get_by_id(id: int) -> dict:
    return {
        "test": "mpla",
        "fileId": id,
    }
