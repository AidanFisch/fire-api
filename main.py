from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import time

from model import run_fire_model  # <-- your big function lives in model.py


# NEW: import budget helpers
from budget import (
    save_month_budget,
    get_month_budget,
    get_year_overview,
    get_series,
    list_all_categories
)


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



---------- NEW: Budget models & endpoints ----------

class ExpenseItem(BaseModel):
    category: str
    planned: float = 0.0
    actual: Optional[float] = None

    @validator("category")
    def _cat_nonempty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("category cannot be empty")
        return v.strip()

    @validator("planned", "actual", pre=True)
    def _num(cls, v):
        if v is None:
            return v
        try:
            return float(v)
        except Exception:
            raise ValueError("Amounts must be numeric")

class BudgetMonthIn(BaseModel):
    month: str                   # "YYYY-MM"
    income_planned: float
    income_actual: Optional[float] = None
    expenses: List[ExpenseItem] = Field(default_factory=list)
    notes: Optional[str] = None
    merge: bool = True           # merge with existing categories by default

    @validator("month")
    def _month_fmt(cls, v: str):
        try:
            import datetime
            datetime.datetime.strptime(v, "%Y-%m")
            return v
        except Exception:
            raise ValueError("month must be 'YYYY-MM'")

    @validator("income_planned", "income_actual", pre=True)
    def _amt(cls, v):
        if v is None:
            return v
        try:
            return float(v)
        except Exception:
            raise ValueError("Income amounts must be numeric")

@app.post("/budget/save_month")
def budget_save_month(req: BudgetMonthIn):
    try:
        out = save_month_budget(
            month=req.month,
            income_planned=req.income_planned,
            income_actual=req.income_actual,
            expenses=[e.dict() for e in req.expenses],
            notes=req.notes,
            merge=req.merge
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/budget/month/{month}")
def budget_month(month: str):
    try:
        return get_month_budget(month)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/budget/year/{year}")
def budget_year(year: int):
    try:
        if year < 1900 or year > 3000:
            raise ValueError("year out of range")
        return get_year_overview(year)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/budget/series")
def budget_series(from_month: str, to_month: str):
    """
    Time series for plotting: planned vs actual net savings and cumulative actual.
    Example: /budget/series?from=2026-01&to=2026-12
    """
    try:
        return get_series(from_month, to_month)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/budget/categories")
def budget_categories():
    try:
        return {"categories": list_all_categories()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

