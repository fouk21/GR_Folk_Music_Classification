import logging
import os
import toml

from datetime import datetime
from fastapi import FastAPI
from lifecycle import lifespan
from routers.dataset import list
from routers.health import health, ping
from routers.uploads import create


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TODAY = datetime.now().strftime('%Y%m%d')


# ------------------------ #
#          Logger          #
# ------------------------ #
api_logger = logging.getLogger('api_logger')
api_logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    '%(asctime)s %(levelname)8s %(process)7d > %(message)s',
    '%Y-%m-%d %H:%M:%S',
)

# File handler
file_handler = logging.FileHandler(
    f'{CURRENT_DIR}/logs/{TODAY}.log', 'a+', 'utf-8'
)
file_handler.setFormatter(formatter)
api_logger.addHandler(file_handler)

# Stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
api_logger.addHandler(stream_handler)

# ----------------------- #
#      Configuration      #
# ----------------------- #
config_file_path = f'{CURRENT_DIR}/../config.toml'
config = toml.load(config_file_path)

# Extract tags metadata from the configuration
tags_metadata = config.get("tags", [])

os.environ['INDEX_PATH'] = str(CURRENT_DIR)


# ------------------------ #
#      FastApi routes      #
# ------------------------ #
app = FastAPI(
    title='Deep Learning - GR Folk Music Classification',
    description='''
        A simple FastAPI application to demonstrate the classification
    ''',
    version="0.1.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)
app.include_router(create.router)
app.include_router(ping.router)
app.include_router(health.router)
app.include_router(list.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
