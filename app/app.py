from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from fastapi.responses import FileResponse
import uvicorn
import tempfile
import json
from triton.client import *

import tritonclient.grpc as grpcclient

app = FastAPI()

@app.get("/")
def healthcheck() -> bool:
    """Check the server's status."""
    return True

@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        # Convert the image to uint8 numpy array
        image = np.frombuffer(contents, dtype=np.uint8)
        # Get the Triton client
        client = get_triton_client()
        # Run the inference
        result_image, postprocess_output = run_inference("yolov5m_ensemble", image, client)
        
        # Save the processed and visualized image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp_path = temp.name
            cv2.imwrite(temp_path, result_image)
        return FileResponse(temp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# uvicorn app:app --host=0.0.0.0 --port=8080

