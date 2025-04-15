'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05
'''

import sys
import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import argparse
import uvicorn

import base64
from typing import Dict, List

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API')
    parser.add_argument('--som_model_path', type=str, default='../../weights/icon_detect/model.pt', help='Path to the som model')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=7860, help='Port for the API')
    args = parser.parse_args()
    return args

args = parse_arguments()
config = vars(args)

app = FastAPI()
omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str

class ParseResponse(BaseModel):
    image: str
    parsed_content_list: str
    label_coordinates: str
    latency: float

@app.post("/parse/", response_model=ParseResponse)
async def parse(
    image_file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
):

    print('start parsing...')
    start = time.time()

    try:
        # Read and encode the image as base64 string
        contents = await image_file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")


    dino_labled_img, label_coordinates, parsed_content_list = omniparser.parse(base64_image)
    latency = time.time() - start

    # Function to format the parsed content into the desired output
    def format_text_boxes(parsed_content_list):
        formatted_output = []
        for idx, item in enumerate(parsed_content_list, start=1):
            # Create the formatted string as 'Text Box ID <number>: content'
            formatted_output.append(f"Text Box ID {idx}: {item['content']}")
        return "\n".join(formatted_output)
    
    # Get the formatted output
    formatted_content = format_text_boxes(parsed_content_list)


    print('time:', latency)
    res = ParseResponse(
            image=dino_labled_img,
            parsed_content_list=str(formatted_content),
            label_coordinates=str(label_coordinates),
            latency=latency
    )
    print(res)
    return res

    # res = {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, "label_coordinates": label_coordinates, "latency": latency}
    # return res

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

if __name__ == "__main__":
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)
