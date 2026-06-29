from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import time
import os
import json
import stripe
import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth
from datetime import date

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# -------- Firebase Admin (server-side only — bypasses Firestore security
# rules, so this is the only place subscription.tier may ever be written) --------
_fb_sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
if _fb_sa_json:
    firebase_admin.initialize_app(credentials.Certificate(json.loads(_fb_sa_json)))
    fs_db = firestore.client()
else:
    fs_db = None
    print("[firebase] FIREBASE_SERVICE_ACCOUNT_JSON not set — subscription writes disabled", flush=True)

def _set_subscription(firebase_uid: Optional[str], data: Dict[str, Any]):
    if not fs_db or not firebase_uid:
        print(f"[stripe] skip subscription write — fs_db={bool(fs_db)} uid={firebase_uid}", flush=True)
        return
    fs_db.collection("profiles").document(firebase_uid).set(
        {"subscription": {**data, "updatedAt": firestore.SERVER_TIMESTAMP}},
        merge=True
    )

# -------- Per-account model-run quota (server-enforced) --------
# A client-side-only counter can be wiped by incognito mode or clearing
# localStorage. Verifying the Firebase ID token and checking/incrementing
# the quota here means the limit actually follows the account, not the
# browser. Anonymous (no token) requests are left ungated — there's no
# account to tie a server-side limit to, and the existing client-side
# check already gives reasonable friction for that case.
FREE_DAILY_RUN_LIMIT = 5

def _get_uid_from_request(request: Request) -> Optional[str]:
    authz = request.headers.get("authorization") or request.headers.get("Authorization")
    if not authz or not authz.startswith("Bearer "):
        return None
    token = authz.split(" ", 1)[1]
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded.get("uid")
    except Exception as e:
        print(f"[auth] token verify failed: {e}", flush=True)
        return None

def _check_and_increment_run_quota(uid: str) -> Optional[str]:
    """Returns None if OK to proceed, or an error detail string if the free
    daily run quota is exceeded. Pro accounts are unlimited."""
    if not fs_db:
        return None
    doc_ref = fs_db.collection("profiles").document(uid)
    snap = doc_ref.get()
    data = snap.to_dict() or {}
    tier = (data.get("subscription") or {}).get("tier")
    if tier == "pro":
        return None
    today = date.today().isoformat()
    usage = data.get("modelRunUsage") or {}
    count = usage.get("count", 0) if usage.get("date") == today else 0
    if count >= FREE_DAILY_RUN_LIMIT:
        return f"Free plan is limited to {FREE_DAILY_RUN_LIMIT} model runs per day. Upgrade for unlimited runs."
    doc_ref.set({"modelRunUsage": {"date": today, "count": count + 1}}, merge=True)
    return None

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
    life_events: List[Dict[str, Any]] = Field(default_factory=list)
    display_month: bool = True

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/fire")
def fire_calc(req: FireRequest, request: Request):
    uid = _get_uid_from_request(request)
    if uid:
        quota_error = _check_and_increment_run_quota(uid)
        if quota_error:
            raise HTTPException(status_code=403, detail=quota_error)

    t0 = time.perf_counter()
    try:
        t1 = time.perf_counter()
        result = run_fire_model(
            inputs=req.inputs,
            property_list=req.property_list,
            life_events=req.life_events,
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
        price_id     = payload.get("price_id")
        email        = payload.get("email")
        firebase_uid = payload.get("firebase_uid")
        success_url  = payload.get("success_url", "https://wealthmodel.io")
        cancel_url   = payload.get("cancel_url",  "https://wealthmodel.io")

        if not price_id:
            raise HTTPException(status_code=400, detail="price_id required")
        if not firebase_uid:
            raise HTTPException(status_code=400, detail="firebase_uid required")

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=success_url + "?checkout_success=true&session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            customer_email=email or None,
            # Lets the webhook map Stripe events back to a Firestore profile
            # without trusting anything the client sends after the fact.
            client_reference_id=firebase_uid,
            subscription_data={"metadata": {"firebase_uid": firebase_uid}},
        )
        return {"url": session.url}
    except stripe.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stripe/cancel-subscription")
def stripe_cancel_subscription(payload: Dict[str, Any] = Body(...)):
    """
    Best-effort immediate cancellation, used when a user deletes their
    account entirely. Always returns 200 — a Stripe-side failure here
    (e.g. subscription already cancelled) shouldn't block account deletion;
    the subscription.deleted webhook is the real source of truth either way.
    """
    sub_id = payload.get("subscription_id")
    if not sub_id:
        return {"cancelled": False, "detail": "no subscription_id provided"}
    try:
        stripe.Subscription.cancel(sub_id)
        return {"cancelled": True}
    except Exception as e:
        print(f"[stripe] cancel-subscription error: {e}", flush=True)
        return {"cancelled": False, "detail": str(e)}

@app.post("/stripe/customer-portal")
def stripe_customer_portal(payload: Dict[str, Any] = Body(...)):
    try:
        sub_id     = payload.get("subscription_id")
        return_url = payload.get("return_url", "https://wealthmodel.io")
        if not sub_id:
            raise HTTPException(status_code=400, detail="subscription_id required")
        sub         = stripe.Subscription.retrieve(sub_id)
        customer_id = getattr(sub, "customer", None)
        if not customer_id:
            raise HTTPException(status_code=400, detail="No customer found for subscription")
        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return {"url": portal.url}
    except stripe.StripeError as e:
        print(f"[stripe] customer-portal StripeError: {e}", flush=True)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"[stripe] customer-portal error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stripe/verify-session")
def stripe_verify_session(session_id: str):
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        paid    = session.payment_status == "paid"
        sub_id  = session.subscription  # string ID (not expanded)
        print(f"[stripe] verify-session paid={paid} sub_id={sub_id}", flush=True)

        # Fetch subscription details safely — skip gracefully if unavailable
        sub_status = None
        cpe        = None
        if sub_id and isinstance(sub_id, str):
            try:
                sub        = stripe.Subscription.retrieve(sub_id)
                sub_status = getattr(sub, "status", None)
                cpe        = getattr(sub, "current_period_end", None)
                print(f"[stripe] sub status={sub_status} cpe={cpe}", flush=True)
            except Exception as sub_err:
                print(f"[stripe] sub fetch error (non-fatal): {sub_err}", flush=True)

        return {
            "paid":               paid,
            "subscription_id":    sub_id,
            "subscription_status":sub_status,
            "current_period_end": cpe,
        }
    except stripe.StripeError as e:
        print(f"[stripe] verify-session StripeError: {e}", flush=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"[stripe] verify-session error: {traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Source of truth for subscription.tier. Stripe signs every event with
    STRIPE_WEBHOOK_SECRET, so this is the only path (besides the Firebase
    console) that can grant Pro — the client can no longer write its own
    subscription field once Firestore rules are tightened.
    """
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.SignatureVerificationError) as e:
        print(f"[stripe] webhook signature error: {e}", flush=True)
        raise HTTPException(status_code=400, detail="Invalid signature")

    etype = event["type"]
    obj   = event["data"]["object"]
    print(f"[stripe] webhook event={etype}", flush=True)

    try:
        if etype == "checkout.session.completed":
            uid         = getattr(obj, "client_reference_id", None)
            sub_id      = getattr(obj, "subscription", None)
            customer_id = getattr(obj, "customer", None)
            status, cpe = None, None
            if sub_id:
                sub    = stripe.Subscription.retrieve(sub_id)
                status = getattr(sub, "status", None)
                cpe    = getattr(sub, "current_period_end", None)
            _set_subscription(uid, {
                "tier":             "pro" if status in ("active", "trialing") else "free",
                "stripeCustomerId": customer_id,
                "subscriptionId":   sub_id,
                "status":           status,
                "currentPeriodEnd": cpe,
            })

        elif etype in ("customer.subscription.updated", "customer.subscription.created"):
            metadata = getattr(obj, "metadata", None)
            uid      = getattr(metadata, "firebase_uid", None) if metadata else None
            status   = getattr(obj, "status", None)
            _set_subscription(uid, {
                "tier":             "pro" if status in ("active", "trialing") else "free",
                "stripeCustomerId": getattr(obj, "customer", None),
                "subscriptionId":   getattr(obj, "id", None),
                "status":           status,
                "currentPeriodEnd": getattr(obj, "current_period_end", None),
            })

        elif etype == "customer.subscription.deleted":
            metadata = getattr(obj, "metadata", None)
            uid      = getattr(metadata, "firebase_uid", None) if metadata else None
            _set_subscription(uid, {
                "tier":             "free",
                "stripeCustomerId": getattr(obj, "customer", None),
                "subscriptionId":   getattr(obj, "id", None),
                "status":           "canceled",
                "currentPeriodEnd": getattr(obj, "current_period_end", None),
            })
    except Exception as e:
        print(f"[stripe] webhook handling error ({etype}): {e}", flush=True)
        raise HTTPException(status_code=500, detail="Webhook handling error")

    return {"received": True}
