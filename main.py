from fastapi import FastAPI
import uvicorn
from qdrant_client import QdrantClient
from pydantic import BaseModel
import utils
from models import ViTForImageClassification, PostData
import torch
import base64
import numpy as np
import matplotlib.pyplot as plt
from typing import List

NUM_CLASSES = 5
IMAGE_WIDTH = IMAGE_HEIGHT = 224
NUM_PATCH_PER_DIM = 16
NUM_LAYERS = 4
NUM_HEADS = 4
PATCH_SIZE = IMAGE_WIDTH // NUM_PATCH_PER_DIM
NUM_PATCH = NUM_PATCH_PER_DIM ** 2
HIDDEN_DIM = 256
INTERMEDIATE_DIM = 512

# Path to your saved .pth file
model = ViTForImageClassification(NUM_CLASSES, NUM_PATCH, PATCH_SIZE, NUM_LAYERS, NUM_HEADS, HIDDEN_DIM, INTERMEDIATE_DIM)
model_path = 'model_ckpt.pth'

# Load the state dictionary into the model
if not torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(model_path))

client = QdrantClient(
    "https://9fb72511-e5fb-4c9f-b96c-67ab52b51d0c.us-east4-0.gcp.cloud.qdrant.io",
    api_key="I18Nhst5caBAc6UQlTkmQv-s-VtZBFCyguyJ4ZWa4OyZXqOE9AOdww",
)
transformations = utils.transform(image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH)
app = FastAPI()


@app.get("/")
def index():
    return "Hello world"


@app.post("/api/get_image")
def get_image(data: PostData):
    result = utils.get_image(img_base64=data.image, model=model, transformations=transformations, client=client)

    return result


