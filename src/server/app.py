import os
import argparse

import uvicorn
from fastapi import Depends, FastAPI, Request
import torch
from extras.loggings import get_logger
from server.server_utils import InferenceArguments, ModelHandler, PredictModel, predict

logger = get_logger("TextClassDeploy")
app = FastAPI()




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='', help="model path")
    parser.add_argument("--pad_size", type=int, default=768, help="pad size")
    parser.add_argument("--num_gpus", type=int, default=0, help="number of GPUs to use")
    parser.add_argument("--port", type=int, default=9090, help="Port to run the server on")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers to run the server")
    args = parser.parse_args()
    logger.info(f"args:{args}")
    return InferenceArguments(**vars(args))

args = get_args()
# 在应用启动时创建模型处理器的单一实例
model_handler = ModelHandler(args)


@app.post("/textclass/")
async def textclass(item: PredictModel,model_data=Depends(model_handler.get_model)):
    try:
        text = item.text
        pad_size = item.pad_size
        model, tokenizer, device = model_data
        prob_np, pred_np = predict(model, tokenizer,text, pad_size, device)  # 转换为numpy数组并确保在CPU
    except Exception as e:
       return {"msg": f"An error occurred while making prediction: {str(e)}"}
    logger.info(f"text:{text}\nprob:{prob_np},pred:{pred_np}")
    # Format the response
    response = {
        "msg": "Prediction made successfully",
        "data": {
            "prob": prob_np,  # Convert to list for JSON serialization
            "pred": pred_np  # Get the predicted class
        }
    }
    return response




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port,workers=args.workers)

