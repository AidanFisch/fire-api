from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# allow your website to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later, replace with your Webflow domain
    allow_methods=["*"],
    allow_headers=["*"]
)

class FireInput(BaseModel):
    current_assets: float
    annual_savings: float
    annual_expenses: float

@app.post("/fire")
def fire_calc(data: FireInput):
    fire_number = data.annual_expenses * 25
    years_to_fire = (fire_number - data.current_assets) / data.annual_savings
    projection = [data.current_assets + data.annual_savings * i for i in range(int(years_to_fire)+1)]
    return {
        "fire_number": fire_number,
        "years_to_fire": years_to_fire,
        "projection": projection
    }
