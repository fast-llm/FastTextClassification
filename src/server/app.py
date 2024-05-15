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
    model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "")
    pad_size = int(os.getenv("PAD_SIZE", 768))
    num_gpus = int(os.getenv("NUM_GPUS", 0))
    port = int(os.getenv("PORT", 9090))
    workers = int(os.getenv("WORKERS", 1))
    args = InferenceArguments(model_name_or_path=model_name_or_path, 
                              pad_size=pad_size, 
                              num_gpus=num_gpus, 
                              port=port, 
                              workers=workers)
    logger.info(f"args:{args}")
    return args

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
    uvicorn.run(app, host="0.0.0.0", port=args.port)