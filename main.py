from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import time
import os
import stripe

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

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
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Existing FIRE endpoint --------
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
        print(
            f"[timing] /fire parse={(t1-t0)*1000:.1f}ms | model={(t2-t1)*1000:.1f}ms | total={(t2-t0)*1000:.1f}ms",
            flush=True
        )
        return result
    except Exception as e:
        print(f"[error] /fire {e}", flush=True)
        raise HTTPException(status_code=400, detail=str(e))

# -------- Budget endpoints (plain-dict style, like model.py) --------
def _require_month_str(m: str) -> str:
    # allow "YYYY-MM" only
    if not isinstance(m, str) or len(m) != 7 or m[4] != "-":
        raise HTTPException(status_code=400, detail="month must be 'YYYY-MM'")
    return m

def _to_float_or_none(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        raise HTTPException(status_code=400, detail="Amounts must be numeric")

@app.post("/budget/save_month")
def budget_save_month(payload: Dict[str, Any] = Body(...)):
    """
    Accept raw dict like model.py does:
    {
      "month": "2026-02",
      "income_planned": 9000,
      "income_actual": 9050,    # optional
      "expenses": [
        {"category": "Housing", "planned": 2500, "actual": 2600},
        ...
      ],
      "notes": "...",
      "merge": true
    }
    """
    try:
        month = _require_month_str(payload.get("month"))
        ip = _to_float_or_none(payload.get("income_planned"))
        if ip is None:
            raise HTTPException(status_code=400, detail="'income_planned' is required")

        ia = _to_float_or_none(payload.get("income_actual"))
        notes = payload.get("notes")
        merge = bool(payload.get("merge", True))

        # normalize expenses to a list of dicts with the three keys
        raw_exp = payload.get("expenses") or []
        if not isinstance(raw_exp, list):
            raise HTTPException(status_code=400, detail="'expenses' must be a list")

        expenses: List[Dict[str, Any]] = []
        for i, e in enumerate(raw_exp):
            if not isinstance(e, dict):
                raise HTTPException(status_code=400, detail=f"expense {i} must be an object")
            cat = (e.get("category") or "").strip()
            if not cat:
                raise HTTPException(status_code=400, detail=f"expense {i} missing 'category'")
            planned = _to_float_or_none(e.get("planned")) or 0.0
            actual = _to_float_or_none(e.get("actual"))
            expenses.append({"category": cat, "planned": planned, "actual": actual})

        return save_month_budget(
            month=month,
            income_planned=ip,
            income_actual=ia,
            expenses=expenses,
            notes=notes,
            merge=merge
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/budget/month/{month}")
def budget_month(month: str):
    try:
        _require_month_str(month)
        return get_month_budget(month)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/budget/year/{year}")
def budget_year(year: int):
    try:
        if year < 1900 or year > 3000:
            raise HTTPException(status_code=400, detail="year out of range")
        return get_year_overview(year)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/budget/series")
def budget_series(from_month: str, to_month: str):
    try:
        _require_month_str(from_month)
        _require_month_str(to_month)
        return get_series(from_month, to_month)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/budget/categories")
def budget_categories():
    try:
        return {"categories": list_all_categories()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------- Stripe endpoints --------

@app.post("/stripe/create-checkout")
def stripe_create_checkout(payload: Dict[str, Any] = Body(...)):
    try:
        price_id    = payload.get("price_id")
        email       = payload.get("email")
        success_url = payload.get("success_url", "https://wealthmodel.io")
        cancel_url  = payload.get("cancel_url",  "https://wealthmodel.io")

        if not price_id:
            raise HTTPException(status_code=400, detail="price_id required")

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=success_url + "?checkout_success=true&session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            customer_email=email or None,
        )
        return {"url": session.url}
    except stripe.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stripe/verify-session")
def stripe_verify_session(session_id: str):
    try:
        session = stripe.checkout.Session.retrieve(
            session_id,
            expand=["subscription"]
        )
        paid = session.payment_status == "paid"
        sub  = session.subscription
        return {
            "paid":                  paid,
            "subscription_id":       sub.id            if sub else None,
            "subscription_status":   sub.status        if sub else None,
            "current_period_end":    sub.current_period_end if sub else None,
        }
    except stripe.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
