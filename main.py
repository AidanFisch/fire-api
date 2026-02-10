from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import time

from model import run_fire_model  # <-- your big function lives in model.py

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aidanfisch.github.io",
        "http://localhost:3000",
        "http://localhost:5173",
        "https://wealthmodel.io",
        "http://wealthmodel.io",
        "https://www.wealthmodel.io",
        "http://www.wealthmodel.io",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FireRequest(BaseModel):
    inputs: Dict[str, Any] = Field(..., description="Full inputs dict from frontend")
    property_list: List[Dict[str, Any]] = Field(default_factory=list)
    display_month: bool = True


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/fire")
def fire_calc(req: FireRequest):
    t0 = time.perf_counter()
    try:
        t1 = time.perf_counter()
        result = run_fire_model(
            inputs=req.inputs,
            property_list=req.property_list,
            display_month=req.display_month
        )
        t2 = time.perf_counter()

        # Log timings to Render logs
        print(f"[timing] /fire parse={(t1-t0)*1000:.1f}ms | model={(t2-t1)*1000:.1f}ms | total={(t2-t0)*1000:.1f}ms", flush=True)

        return result
    except Exception as e:
        print(f"[error] /fire {e}", flush=True)
        raise HTTPException(status_code=400, detail=str(e))

