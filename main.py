from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from model import run_fire_model  # <-- your big function lives in model.py

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aidanfisch.github.io",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FireRequest(BaseModel):
    inputs: Dict[str, Any] = Field(..., description="Full inputs dict from frontend")
    property_list: List[Dict[str, Any]] = Field(..., description="List of property dicts from frontend")
    display_month: bool = True

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/fire")
def fire_calc(req: FireRequest):
    try:
        result = run_fire_model(
            inputs=req.inputs,
            property_list=req.property_list,
            display_month=req.display_month
        )
        return result
    except Exception as e:
        # bubble up a clean error for debugging frontend
        raise HTTPException(status_code=400, detail=str(e))
