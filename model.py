


  # =========================
  # INPUTS
  # =========================

inputs_default = {
    "current_age": 25,
    "inflation": 0.025,
    "todays_lifestyle_income": 80000,
    "initial_savings": 500000,
    "current_income": 200000,
    "current_expenses": 64000,
    "average_salary_increase": 0.035,
    "end_age": 60,
    "stock_growth": 0.06,
    "stock_yearly_contribution": 20000,
    "starting_stock_value": 35000,
    # ---- SUPER (NEW)
    "super_starting_balance": 120000,
    "super_sg_rate": 0.11,                 # SG rate (11% from 2025)
    "super_growth": 0.065,                 # long-term return assumption
    "super_additional_annual": 0,          # voluntary extra, $/yr (affects cash)
    "fire_swr": 0.04,          # 4% rule
    "super_access_age": 60,    # when super can be drawn
}

property_1_input = {
    "id": 1,
    "name": "Apartment",
    "purchase_price": 400000,
    "current_value": 640000,
    "original_loan": 325000,
    "loan_balance_current": 215000,
    "purchase_fees": 20000,
    "monthly_rent": 2400,
    "strata_quarterly": 1200,
    "rates_quarterly": 900,
    "other_costs_monthly": 200,
    "interest_rate": 0.055,
    "property_growth": 0.03,
    "rental_growth": 0.03,
    "year_bought": 2021,
    "loan_term_years": 30,
    "use_offset": True,
    "is_owner_occupied": False

}

# property_2_input = {
#     "id": 2,
#     "name": "House",
#     "purchase_price": 1600000,
#     "current_value": 1600000,
#     "original_loan": 1280000,
#     "loan_balance_current": 1280000,
#     "purchase_fees": 80000,
#     "monthly_rent": 5333,
#     "strata_quarterly": 0,
#     "rates_quarterly": 0,
#     "other_costs_monthly": 0,
#     "interest_rate": 0.055,
#     "property_growth": 0.03,
#     "rental_growth": 0.03,
#     "year_bought": 2026,
#     "loan_term_years": 30,
#     "use_offset": True,
#     "is_owner_occupied": False

# }

property_3_input = {
    "id": 3,
    "name": "PPOR",
    "purchase_price": 1200000,
    "current_value": 1200000,
    "original_loan": 900000,
    "loan_balance_current": 0,   # if buying in future
    "purchase_fees": 40000,
    "monthly_rent": 0,           # MUST be 0
    "strata_quarterly": 0,        # optional
    "rates_quarterly": 900,
    "other_costs_monthly": 300,
    "interest_rate": 0.055,
    "property_growth": 0.03,
    "rental_growth": 0.0,         # MUST be 0
    "year_bought": 2027,
    "loan_term_years": 30,
    "use_offset": True,

    # 🔥 NEW
    "is_owner_occupied": True,
}

# property_list_default = [property_1_input, property_2_input, property_3_input]
property_list_default = [property_1_input, property_3_input]
# display_month = False




from typing import Optional

def run_fire_model(inputs: dict | None, property_list: list | None, display_month: bool = True):
    import pandas as pd
    import numpy as np
    from datetime import date

    REQUIRED_PROP_KEYS = {
    "id","name","purchase_price","current_value","original_loan","loan_balance_current",
    "purchase_fees","monthly_rent","strata_quarterly","rates_quarterly","other_costs_monthly",
    "interest_rate","property_growth","rental_growth","year_bought","loan_term_years",
    "use_offset","is_owner_occupied"
}


    # Fallbacks for notebook testing
    if inputs is None:
        inputs = inputs_default
    if property_list is None:
        property_list = []

    ###debugging###
    required_input_keys = set(inputs_default.keys())
    missing = required_input_keys - set(inputs.keys())
    if missing:
        raise ValueError(f"Missing inputs keys: {sorted(missing)}")

    # validate properties have required keys too
    required_prop_keys = set().union(*(p.keys() for p in property_list_default))

    for idx, p in enumerate(property_list):
        missing_p = REQUIRED_PROP_KEYS - set(p.keys())
        if missing_p:
            raise ValueError(f"Property[{idx}] missing keys: {sorted(missing_p)}")





    # =========================
    # TAX (annual)
    # =========================

    def tax_payable(income):
        """2025-26 marginal rates, no Medicare etc (add later if you want)."""
        if income <= 18200:
            return 0
        elif income <= 45000:
            return (income - 18200) * 0.19
        elif income <= 120000:
            return 5092 + (income - 45000) * 0.325
        elif income <= 180000:
            return 29467 + (income - 120000) * 0.37
        else:
            return 51667 + (income - 180000) * 0.45

    # =========================
    # HELPERS
    # =========================

    def monthly_rate(annual_rate: float) -> float:
        return annual_rate / 12.0

    def monthly_growth_factor(annual_growth: float) -> float:
        """Convert annual growth to equivalent monthly multiplicative factor."""
        return (1 + annual_growth) ** (1/12)

    def pmt_from_balance(balance, annual_rate, remaining_months):
        """Monthly payment for remaining term, given current balance."""
        if balance <= 0:
            return 0.0
        r = monthly_rate(annual_rate)
        n = max(int(remaining_months), 1)
        return balance * r / (1 - (1 + r) ** (-n))

    def months_between_years(start_year, end_year):
        return (end_year - start_year) * 12

    # =========================
    # MODEL HORIZON (MONTHLY)
    # =========================

    start_year = date.today().year
    end_year = start_year + (inputs["end_age"] - inputs["current_age"]) - 1
    months = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq="MS")

    # Build base monthly dataframe
    dfm = pd.DataFrame({"Date": months})
    dfm["Year"] = dfm["Date"].dt.year
    dfm["Month"] = dfm["Date"].dt.month
    dfm["t_months"] = range(len(dfm))  # 0..N
    dfm["Age"] = inputs["current_age"] + dfm["t_months"] / 12.0

    # Salary/expenses grow with YEARS FROM NOW (not "i from birth")
    dfm["t_years"] = (dfm["t_months"]+1) / 12.0

    # Use annual compounding continuously via (1+g)^(t_years)
    dfm["Salary_Annual"] = inputs["current_income"] * ((1 + inputs["average_salary_increase"]) ** dfm["t_years"])
    dfm["Expenses_Annual"] = inputs["current_expenses"] * ((1 + inputs["inflation"]) ** dfm["t_years"])

    dfm["Salary_Monthly"] = dfm["Salary_Annual"] / 12.0
    dfm["Expenses_Monthly"] = dfm["Expenses_Annual"] / 12.0

    dfm["Target_Annual_Infl_Adj"] = inputs["todays_lifestyle_income"] * ((1 + inputs["inflation"]) ** dfm["t_years"])
    dfm["Target_Monthly_Infl_Adj"] = dfm["Target_Annual_Infl_Adj"] / 12.0


    # Stocks monthly
    stock_m_growth = monthly_growth_factor(inputs["stock_growth"]) - 1
    dfm["Stock_Contribution_Annual"] = inputs["stock_yearly_contribution"] * ((1 + inputs["inflation"]) ** dfm["t_years"])
    dfm["Stock_Contribution_Monthly"] = dfm["Stock_Contribution_Annual"] / 12.0

    # ---- Super monthly
    super_m_growth = monthly_growth_factor(inputs["super_growth"]) - 1

    # voluntary contribution (annual -> monthly)
    dfm["Super_Extra_Annual"] = float(inputs.get("super_additional_annual", 0.0)) * ((1 + inputs["inflation"]) ** dfm["t_years"])
    dfm["Super_Extra_Monthly"] = dfm["Super_Extra_Annual"] / 12.0




    def allocate_weighted_offset(cash_pool: float, balances: dict[str, float]) -> dict[str, float]:
        """
        Allocate cash_pool across loans proportionally to balances, with caps (<= balance).
        Redistributes leftover cash if some loans cap out.
        balances: {loan_name: balance} for offset-enabled loans only.
        Returns: {loan_name: allocated_offset}
        """
        # Clean inputs
        cash_pool = float(max(cash_pool, 0.0))
        balances = {k: float(max(v, 0.0)) for k, v in balances.items() if v is not None and v > 0}

        alloc = {k: 0.0 for k in balances}
        if cash_pool <= 0 or not balances:
            return alloc

        remaining_cash = cash_pool
        remaining = balances.copy()

        # Iterate because caps can cause leftover that must be re-allocated
        while remaining_cash > 1e-9 and remaining:
            total_bal = sum(remaining.values())
            if total_bal <= 0:
                break

            # Pro-rata allocation
            provisional = {k: remaining_cash * (bal / total_bal) for k, bal in remaining.items()}

            # Apply caps and compute leftover
            new_remaining = {}
            leftover = 0.0
            for k, prov in provisional.items():
                cap = remaining[k]
                used = min(prov, cap)
                alloc[k] += used
                if prov > cap:
                    leftover += (prov - cap)  # this amount couldn't be placed here
                else:
                    # still has capacity left after placing used
                    new_remaining[k] = cap - used

            remaining_cash = leftover
            remaining = {k: v for k, v in new_remaining.items() if v > 1e-9}

            # If nothing changed (numerical edge), stop
            if leftover <= 1e-9:
                break

        return alloc


    # def compute_monthly_interest(balance: float, offset_alloc: float, annual_rate: float) -> float:
    #     net = max(float(balance) - float(offset_alloc), 0.0)
    #     return net * float(annual_rate) / 12.0

    # =========================
    # PROPERTY STATE INITIALISATION (monthly engine)
    # =========================

    # We assume settlements happen in January of year_bought. If you want a specific month, add "month_bought".
    def purchase_month(year_bought: int) -> pd.Timestamp:
        return pd.Timestamp(f"{year_bought}-01-01")

    # Property state containers
    prop_state = {}
    for p in property_list:
        name = p["name"].replace(" ", "_")
        prop_state[name] = {
            "active": False,
            "loan_balance": 0.0,
            "property_value": 0.0,
            "rent_monthly": 0.0,
            "monthly_pmt": 0.0,
            "remaining_months": 0,
        }

    # If property already owned before start_year, initialise as active at start
    for p in property_list:
        name = p["name"].replace(" ", "_")
        if p["year_bought"] < start_year:
            # treat as active from start
            prop_state[name]["active"] = True
            prop_state[name]["loan_balance"] = float(p["loan_balance_current"])
            prop_state[name]["property_value"] = float(p["current_value"])
            prop_state[name]["rent_monthly"] = float(p["monthly_rent"])

            elapsed = months_between_years(p["year_bought"], start_year)  # assumes bought Jan year_bought
            total_term = int(p["loan_term_years"] * 12)
            remaining = max(total_term - elapsed, 1)
            prop_state[name]["remaining_months"] = remaining
            prop_state[name]["monthly_pmt"] = pmt_from_balance(
                balance=prop_state[name]["loan_balance"],
                annual_rate=p["interest_rate"],
                remaining_months=remaining
            )

    # =========================
    # SIMULATION LOOP (MONTHLY)
    # =========================

    cash_balance = float(inputs["initial_savings"])  # this is your offset/savings bucket
    stock_balance = float(inputs["starting_stock_value"])
    super_balance = float(inputs.get("super_starting_balance", 0.0))




    # Create columns per property
    for p in property_list:
        prefix = p["name"].replace(" ", "_")
        for col in [
            "Mortgage_Balance_Start",
            "Mortgage_Balance",
            "Monthly_Mortgage_PMT",
            "Mortgage_Interest",
            "Mortgage_Principal",
            "Property_Value",
            "Rent_Income",
            "Strata",
            "Rates",
            "Other_Costs",
            "Offset_Allocated",
            "Net_Rent",
            "Purchase_Cashflow",
        ]:
            dfm[f"{prefix}_{col}"] = 0.0
        dfm[f"{prefix}_Active"] = 0   # ✅ add this

    for p in property_list:
      prefix = p["name"].replace(" ", "_")
      dfm[f"{prefix}_Is_PPOR"] = int(p.get("is_owner_occupied", False))

    dfm["Total_Net_Rent"] = 0.0
    dfm["Tax_Paid"] = 0.0
    dfm["Cash_Balance_End"] = 0.0
    dfm["Stock_Balance_End"] = 0.0
    dfm["Stock_Growth_Accrued"] = 0.0
    # ---- Super columns
    dfm["Super_Balance_End"] = 0.0
    dfm["Super_Growth_Accrued"] = 0.0
    dfm["Super_SG_Contribution"] = 0.0
    dfm["Super_Extra_Contribution"] = 0.0

    dfm["Offset_Eligible_Balance_Total"] = 0.0
    dfm["Offset_Allocated_Total"] = 0.0
    dfm["Offset_Unused_Cash"] = 0.0
    dfm["Total_Property_Value"] = 0.0
    dfm["Total_Mortgage_Balance"] = 0.0
    dfm["Net_Worth_Ex_PPOR"] = 0.0  # optional but useful
    dfm["Net_Worth_Incl_PPOR"] = 0.0  # optional but useful
    dfm["Total_Property_Equity"] = 0.0
    dfm["Total_Property_Equity_Ex_PPOR"] = 0.0
    dfm["Total_Property_Value_Ex_PPOR"] = 0.0
    dfm["Total_Mortgage_Balance_Ex_PPOR"] = 0.0
    dfm["Net_Worth_Incl_Super_Incl_PPOR"] = 0.0
    dfm["Net_Worth_Incl_Super_Ex_PPOR"] = 0.0

    dfm["Stock_Drawdown_Monthly"] = 0.0
    dfm["Super_Drawdown_Monthly"] = 0.0
    dfm["FIRE_Income_Monthly"] = 0.0
    dfm["FIRE_Eligible"] = 0
    dfm["FIRE_Age"] = np.nan
    dfm["Fire_Replacement_Cashflow_Monthly"] = 0.0
    dfm["Fire_Principal_Service_Monthly"] = 0.0
    dfm["Fire_Target_Monthly"] = 0.0
    dfm["Fire_Gap_Monthly"] = 0.0

    dfm["Salary_Paid"] = 0.0
    
    dfm["Stock_Contrib_Paid"] = 0.0
    dfm["Expenses_Paid"] = 0.0
    


    fired = False
    fire_age = None

    for i, row in dfm.iterrows():
        dt = row["Date"]
        year = int(row["Year"])
        month = int(row["Month"])


        # ---- Stock update (monthly)
        stock_growth_amt = stock_balance * stock_m_growth
        stock_contrib = 0.0 if fired else row["Stock_Contribution_Monthly"]
        stock_balance += stock_growth_amt + stock_contrib


        dfm.loc[i, "Stock_Growth_Accrued"] = stock_growth_amt
        dfm.loc[i, "Stock_Balance_End"] = stock_balance

        # ---- Income/expenses (monthly)
        salary_m_raw = row["Salary_Monthly"]
        salary_m = salary_m_raw if not fired else 0.0

        expenses_m = row["Expenses_Monthly"]

        stock_contrib_m = 0.0 if fired else stock_contrib
        


        dfm.loc[i, "Salary_Paid"] = salary_m
        

        dfm.loc[i, "Stock_Contrib_Paid"] = stock_contrib


        # ---- Super update (monthly)
        # ---- Super update (monthly)
        super_sg_m = 0.0 if fired else (salary_m * float(inputs.get("super_sg_rate", 0.0)))  # ✅ SG stops too
        super_extra_m = float(row.get("Super_Extra_Monthly", 0.0))                            # voluntary still optional


        super_growth_amt = super_balance * super_m_growth
        super_balance += super_growth_amt + super_sg_m + super_extra_m

        dfm.loc[i, "Super_Growth_Accrued"] = super_growth_amt
        dfm.loc[i, "Super_SG_Contribution"] = super_sg_m
        dfm.loc[i, "Super_Extra_Contribution"] = super_extra_m
        dfm.loc[i, "Super_Balance_End"] = super_balance


        # ---- Activate purchases (if year_bought == this month)
        purchase_cashflow_total = 0.0
        for p in property_list:
            prefix = p["name"].replace(" ", "_")
            buy_dt = purchase_month(p["year_bought"])

            if dt == buy_dt and p["year_bought"] >= start_year:
                # Cash needed at purchase: deposit + fees = purchase_price + fees - loan
                cash_needed = (p["purchase_price"] + p["purchase_fees"] - p["original_loan"])
                purchase_cashflow_total -= cash_needed

                # Initialise property at purchase
                prop_state[prefix]["active"] = True
                prop_state[prefix]["loan_balance"] = float(p["original_loan"])
                # ✅ ensure the "start" value is not left at 0 for this year
                dfm.loc[i, f"{prefix}_Mortgage_Balance_Start"] = prop_state[prefix]["loan_balance"]
                dfm.loc[i, f"{prefix}_Mortgage_Balance"] = prop_state[prefix]["loan_balance"]  # optional but helpful

                prop_state[prefix]["property_value"] = float(p["purchase_price"])
                prop_state[prefix]["rent_monthly"] = float(p["monthly_rent"])

                remaining = int(p["loan_term_years"] * 12)
                prop_state[prefix]["remaining_months"] = remaining
                prop_state[prefix]["monthly_pmt"] = pmt_from_balance(
                    balance=prop_state[prefix]["loan_balance"],
                    annual_rate=p["interest_rate"],
                    remaining_months=remaining
                )

                dfm.loc[i, f"{prefix}_Purchase_Cashflow"] = -cash_needed
                # Apply purchase cashflow immediately so offset reflects reality this month
        cash_balance += purchase_cashflow_total

        # # ---- Offset allocation (use cash available at start-of-month AFTER purchases)
        available_offset_cash = max(cash_balance, 0.0)

        eligible_balances = {}
        for p in property_list:
            prefix = p["name"].replace(" ", "_")
            if prop_state[prefix]["active"] and p.get("use_offset", False):
                bal = float(max(prop_state[prefix]["loan_balance"], 0.0))
                if bal > 0:
                    eligible_balances[prefix] = bal

        offset_allocations = allocate_weighted_offset(available_offset_cash, eligible_balances)
        
        dfm.loc[i, "Offset_Eligible_Balance_Total"] = sum(eligible_balances.values())
        dfm.loc[i, "Offset_Allocated_Total"] = sum(offset_allocations.values())
        dfm.loc[i, "Offset_Unused_Cash"] = max(available_offset_cash - dfm.loc[i, "Offset_Allocated_Total"], 0.0)



        # ---- Offset allocation base: only positive cash can offset
        # Determine total active mortgage balance for pro-rata allocation



        # offset_allocations = {p["name"].replace(" ", "_"): 0.0 for p in property_list}

        


        # ---- Property monthly mechanics
        total_owner_costs = 0.0
        total_net_rent = 0.0
        total_interest = 0.0
        total_principal = 0.0
        total_pmt = 0.0
        total_running_costs = 0.0

        for p in property_list:
            prefix = p["name"].replace(" ", "_")
            if not prop_state[prefix]["active"]:
                dfm.loc[i, f"{prefix}_Mortgage_Balance_Start"] = 0.0
                dfm.loc[i, f"{prefix}_Mortgage_Balance"] = 0.0
                continue

            dfm.loc[i, f"{prefix}_Active"] = 1 
            loan_bal = prop_state[prefix]["loan_balance"]

            dfm.loc[i, f"{prefix}_Mortgage_Balance_Start"] = loan_bal

            if loan_bal <= 0:
                # loan is paid off, but the property is still active and still has:
                # - value growth
                # - rent (if IP)
                # - running costs
                # so we must keep writing those columns each month

                prop_state[prefix]["loan_balance"] = 0.0

                # keep compounding value + rent
                prop_state[prefix]["property_value"] *= monthly_growth_factor(p["property_growth"])
                if not p.get("is_owner_occupied", False):
                    prop_state[prefix]["rent_monthly"] *= monthly_growth_factor(p["rental_growth"])

                strata_m = (p["strata_quarterly"] * 4) / 12.0
                rates_m  = (p["rates_quarterly"] * 4) / 12.0
                other_m  = p["other_costs_monthly"]

                # paid off => no interest/principal/payment
                interest_m = 0.0
                principal_m = 0.0
                actual_pmt_m = 0.0
                offset_alloc = 0.0

                if p.get("is_owner_occupied", False):
                    owner_cost_m = strata_m + rates_m + other_m
                    net_rent_m = 0.0
                else:
                    owner_cost_m = 0.0
                    net_rent_m = prop_state[prefix]["rent_monthly"] - strata_m - rates_m - other_m

                # write outputs so yearly rollup "last"/"sum" never becomes 0
                dfm.loc[i, f"{prefix}_Mortgage_Balance"] = 0.0
                dfm.loc[i, f"{prefix}_Monthly_Mortgage_PMT"] = 0.0
                dfm.loc[i, f"{prefix}_Mortgage_Interest"] = 0.0
                dfm.loc[i, f"{prefix}_Mortgage_Principal"] = 0.0
                dfm.loc[i, f"{prefix}_Property_Value"] = prop_state[prefix]["property_value"]
                dfm.loc[i, f"{prefix}_Rent_Income"] = prop_state[prefix]["rent_monthly"]
                dfm.loc[i, f"{prefix}_Strata"] = strata_m
                dfm.loc[i, f"{prefix}_Rates"] = rates_m
                dfm.loc[i, f"{prefix}_Other_Costs"] = other_m
                dfm.loc[i, f"{prefix}_Offset_Allocated"] = 0.0
                dfm.loc[i, f"{prefix}_Net_Rent"] = net_rent_m

                total_owner_costs += owner_cost_m
                total_net_rent += net_rent_m
                total_interest += 0.0
                total_principal += 0.0
                total_pmt += 0.0
                total_running_costs += (strata_m + rates_m + other_m)

                continue



            # Grow property value + rent monthly
            prop_state[prefix]["property_value"] *= monthly_growth_factor(p["property_growth"])
            if not p.get("is_owner_occupied", False):
              prop_state[prefix]["rent_monthly"] *= monthly_growth_factor(p["rental_growth"])


            # Monthly property running costs
            strata_m = (p["strata_quarterly"] * 4) / 12.0  # quarterly -> annual -> /12
            rates_m = (p["rates_quarterly"] * 4) / 12.0
            other_m = p["other_costs_monthly"]

            # Pro-rata offset allocation by current balance
            # Weighted offset allocation (only for offset-enabled loans)
            offset_alloc = 0.0
            if p.get("use_offset", False):
                offset_alloc = float(offset_allocations.get(prefix, 0.0))
                offset_alloc = min(offset_alloc, loan_bal)
                offset_alloc = max(offset_alloc, 0.0)






            # Mortgage calc (monthly)
            r_m = monthly_rate(p["interest_rate"])
            pmt_m = prop_state[prefix]["monthly_pmt"]


            # Mortgage calc (monthly)
            r_m = monthly_rate(p["interest_rate"])
            pmt_m = prop_state[prefix]["monthly_pmt"]

            interest_m = max(loan_bal - offset_alloc, 0.0) * r_m

            # repayment still happens even if interest is 0
            principal_m = min(max(pmt_m - interest_m, 0.0), loan_bal)
            actual_pmt_m = interest_m + principal_m


            loan_bal_end = loan_bal - principal_m
            assert abs((loan_bal - principal_m) - loan_bal_end) < 1e-6
            prop_state[prefix]["loan_balance"] = loan_bal_end

            # Decrement remaining months (not strictly needed, but keeps PMT logic sane if you later refinance)
            prop_state[prefix]["remaining_months"] = max(prop_state[prefix]["remaining_months"] - 1, 0)

            # Net rent (cashflow): rent - interest - costs (principal is not an expense, it's repayment of balance)
            if p.get("is_owner_occupied", False):
                owner_cost_m = strata_m + rates_m + other_m
                # PPOR: no rent, no deductible costs
                net_rent_m = 0.0
            else:
                # Investment property
                owner_cost_m = 0.0
                net_rent_m = (
                    prop_state[prefix]["rent_monthly"]
                    - interest_m
                    - strata_m
                    - rates_m
                    - other_m
                )


            # Write row outputs
            
            dfm.loc[i, f"{prefix}_Mortgage_Balance"] = loan_bal_end
            dfm.loc[i, f"{prefix}_Monthly_Mortgage_PMT"] = actual_pmt_m
            dfm.loc[i, f"{prefix}_Mortgage_Interest"] = interest_m
            dfm.loc[i, f"{prefix}_Mortgage_Principal"] = principal_m
            dfm.loc[i, f"{prefix}_Property_Value"] = prop_state[prefix]["property_value"]
            dfm.loc[i, f"{prefix}_Rent_Income"] = prop_state[prefix]["rent_monthly"]
            dfm.loc[i, f"{prefix}_Strata"] = strata_m
            dfm.loc[i, f"{prefix}_Rates"] = rates_m
            dfm.loc[i, f"{prefix}_Other_Costs"] = other_m
            dfm.loc[i, f"{prefix}_Offset_Allocated"] = min(offset_alloc, loan_bal_end)
            # dfm.loc[i, f"{prefix}_Offset_Allocated"] = min(offset_alloc, loan_bal)

            # dfm.loc[i, f"{prefix}_Offset_Allocated"] = offset_alloc
            dfm.loc[i, f"{prefix}_Net_Rent"] = net_rent_m

            # dfm[f"{prefix}_Is_PPOR"] = int(p.get("is_owner_occupied", False))

            total_owner_costs += owner_cost_m
            total_net_rent += net_rent_m
            total_interest += interest_m
            total_principal += principal_m
            total_pmt += actual_pmt_m
            total_running_costs += (strata_m + rates_m + other_m)

        dfm.loc[i, "Total_Net_Rent"] = total_net_rent


        # =========================
        # FIRE VIABILITY CHECK (CASHFLOW-BASED)
        # =========================



        # Allowed deterministic drawdowns (DO NOT mutate balances here)
        stock_draw_m = (stock_balance * inputs["fire_swr"]) / 12.0

        super_draw_m = 0.0
        if row["Age"] >= inputs["super_access_age"]:
            super_draw_m = (super_balance * inputs["fire_swr"]) / 12.0

        # Mortgage principal MUST be serviceable post-FIRE
        principal_service_m = total_principal

        taxable_fire_income_m = total_net_rent

        # Stock drawdown – assume 50% CGT discount
        taxable_fire_income_m = total_net_rent + (stock_draw_m * 0.5)
        # (super ignored / assumed tax-free)
        fire_annual_tax = tax_payable(max(taxable_fire_income_m * 12.0, 0.0))  # optional floor at 0 for the tax function
        fire_tax_m = fire_annual_tax / 12.0

        



        # Replacement income AFTER FIRE (salary replacement)
        fire_replacement_cashflow_m = (
            total_net_rent
            + stock_draw_m
            + super_draw_m
            - principal_service_m
            - fire_tax_m

        )

        target_m = row["Target_Monthly_Infl_Adj"]

        fire_gap_m = fire_replacement_cashflow_m - target_m

        dfm.loc[i, "Stock_Drawdown_Monthly"] = stock_draw_m
        dfm.loc[i, "Super_Drawdown_Monthly"] = super_draw_m
        dfm.loc[i, "FIRE_Income_Monthly"] = fire_replacement_cashflow_m


        dfm.loc[i, "Fire_Replacement_Cashflow_Monthly"] = fire_replacement_cashflow_m
        dfm.loc[i, "Fire_Principal_Service_Monthly"] = principal_service_m
        dfm.loc[i, "Fire_Target_Monthly"] = target_m 
        dfm.loc[i, "Fire_Gap_Monthly"] = fire_gap_m

        eligible_now = fire_gap_m >= 0

        if (not fired) and eligible_now:
            fired = True
            fire_age = float(row["Age"])

        dfm.loc[i, "FIRE_Eligible"] = 1 if fired else 0
        dfm.loc[i, "FIRE_Age"] = fire_age if fire_age is not None else np.nan

        # =========================
        # ACTUAL ASSET WITHDRAWALS (POST-FIRE ONLY)
        # =========================

        stock_withdraw_m = 0.0
        super_withdraw_m = 0.0

        if fired:
            stock_withdraw_m = stock_draw_m
            stock_balance -= stock_withdraw_m

            if row["Age"] >= inputs["super_access_age"]:
                super_withdraw_m = super_draw_m
                super_balance -= super_withdraw_m


        total_prop_value = 0.0
        total_mort_bal = 0.0
        ppor_value = 0.0
        ppor_mort = 0.0

        for p in property_list:
            prefix = p["name"].replace(" ", "_")
            if not prop_state[prefix]["active"]:
                continue

            v = float(prop_state[prefix]["property_value"])
            b = float(max(prop_state[prefix]["loan_balance"], 0.0))

            total_prop_value += v
            total_mort_bal += b

            if p.get("is_owner_occupied", False):
                ppor_value += v
                ppor_mort += b

        # ✅ Write totals
        dfm.loc[i, "Total_Property_Value"] = total_prop_value
        dfm.loc[i, "Total_Mortgage_Balance"] = total_mort_bal

        # ✅ Add these totals (equity + ex-PPOR breakdown)
        dfm.loc[i, "Total_Property_Equity"] = max(total_prop_value - total_mort_bal, 0.0)

        dfm.loc[i, "Total_Property_Value_Ex_PPOR"] = max(total_prop_value - ppor_value, 0.0)
        dfm.loc[i, "Total_Mortgage_Balance_Ex_PPOR"] = max(total_mort_bal - ppor_mort, 0.0)
        dfm.loc[i, "Total_Property_Equity_Ex_PPOR"] = max(
            dfm.loc[i, "Total_Property_Value_Ex_PPOR"] - dfm.loc[i, "Total_Mortgage_Balance_Ex_PPOR"],
            0.0
        )


        # ---- TAXABLE INCOME (allow negative gearing)
        taxable_property_income_m = total_net_rent   # can be negative now (negative gearing)
        taxable_stock_income_m = 0.0 if not fired else (stock_draw_m * 0.5)  # 50% CGT discount assumption
        taxable_super_income_m = 0.0  # assume tax-free after access age (your assumption)

        taxable_income_m = taxable_property_income_m + taxable_stock_income_m + taxable_super_income_m

        # Salary adds to taxable income while working
        taxable_income_m += salary_m

        annualised_taxable_income = taxable_income_m * 12.0
        estimated_annual_tax = tax_payable(annualised_taxable_income)

        tax_paid_this_month = estimated_annual_tax / 12.0
        dfm.loc[i, "Tax_Paid"] = tax_paid_this_month






        # # ---- PAYG SMOOTHED TAX (monthly withholding)

        # taxable_fire_income_m = max(taxable_fire_income_m, 0.0)


        # # Annualise current run-rate income
        # annualised_taxable_income = (salary_m + taxable_fire_income_m) * 12.0


        # # Estimate full-year tax
        # estimated_annual_tax = tax_payable(annualised_taxable_income)

        # # Pay 1/12 each month
        # tax_paid_this_month = estimated_annual_tax / 12.0


        # ---- Taxable income accrues monthly (salary + net rent). Tax paid once at December.




        # ---- Cash balance update (THIS is your offset pool)
        # Cashflow rule:
        # cash += salary - expenses - stock_contrib + net_rent - principal - tax + purchase_cashflow
        # (interest is already subtracted inside net_rent, so we do NOT subtract interest again)
        cash_balance = (
            cash_balance
            + salary_m
            - expenses_m
            - stock_contrib_m
            - super_extra_m
            + total_net_rent
            - total_owner_costs
            - total_principal
            - tax_paid_this_month
            
        )

                # =========================




        # ---- Offset allocation AFTER true cash balance is known
#         available_offset_cash = max(cash_balance, 0.0)

#         eligible_balances = {}
#         for p in property_list:
#             prefix = p["name"].replace(" ", "_")
#             if prop_state[prefix]["active"] and p.get("use_offset", False):
#                 bal = float(max(prop_state[prefix]["loan_balance"], 0.0))
#                 if bal > 0:
#                     eligible_balances[prefix] = bal




#         offset_allocations = allocate_weighted_offset(
#             available_offset_cash,
#             eligible_balances
#         )

#         dfm.loc[i, "Offset_Eligible_Balance_Total"] = sum(eligible_balances.values())
#         dfm.loc[i, "Offset_Allocated_Total"] = sum(offset_allocations.values())
#         dfm.loc[i, "Offset_Unused_Cash"] = max(
#             available_offset_cash - dfm.loc[i, "Offset_Allocated_Total"], 0.0
# )


        dfm.loc[i, "Cash_Balance_End"] = cash_balance
                # ---- Net worth (end-of-month)
        dfm.loc[i, "Net_Worth_Incl_PPOR"] = cash_balance + stock_balance + total_prop_value - total_mort_bal
        dfm.loc[i, "Net_Worth_Ex_PPOR"] = (
          cash_balance
          + stock_balance
          + (total_prop_value - ppor_value)
          - (total_mort_bal - ppor_mort)
)
        
        dfm.loc[i, "Net_Worth_Incl_Super_Incl_PPOR"] = cash_balance + stock_balance + super_balance + total_prop_value - total_mort_bal
        dfm.loc[i, "Net_Worth_Incl_Super_Ex_PPOR"] = (
          cash_balance
          + stock_balance
          + super_balance
          + (total_prop_value - ppor_value)
          - (total_mort_bal - ppor_mort)
)



    ### toggle display

    if display_month == True:

      # Round numeric columns only
      # Round money columns only (keep time columns accurate)
      do_not_round = {"t_months", "t_years", "Age", "Year", "Month"}
      numeric_cols = [c for c in dfm.select_dtypes(include=["number"]).columns if c not in do_not_round]
      dfm[numeric_cols] = dfm[numeric_cols].round(0)

      dfm["Age"] = dfm["Age"].round(2)
      dfm["t_years"] = dfm["t_years"].round(3)

      # print(dfm.to_csv(index=False))


    else:
      # =========================
      # ROLL UP TO YEARLY REPORT (end-of-year)
      # dfm["Salary_Paid"] = dfm["Salary_Monthly"]


      dfm["Expenses_Paid"] = dfm["Expenses_Monthly"]
      # =========================

      # Define aggregation: sums for flows, last for balances/values
      agg = {
          "Age": "first",                      # (see note below)
          "Salary_Annual": "last",
          "Expenses_Annual": "last",
          "Salary_Monthly": "last",           # run-rate monthly at end of year
          "Expenses_Monthly": "last",
          "Stock_Contribution_Annual": "last",
          "Stock_Contribution_Monthly": "last",
          "Salary_Paid": "sum",
          
          "Expenses_Paid": "sum",             # ✅ total paid
          "Stock_Contrib_Paid": "sum",        # ✅ total invested
          "Stock_Growth_Accrued": "sum",
          "Stock_Balance_End": "last",
          "Super_Growth_Accrued": "sum",
          "Super_SG_Contribution": "sum",
          "Super_Extra_Contribution": "sum",
          "Super_Balance_End": "last",
          "Net_Worth_Incl_Super_Incl_PPOR": "last",
          "Net_Worth_Incl_Super_Ex_PPOR": "last",
          "Total_Net_Rent": "sum",
          "Tax_Paid": "sum",
          "Cash_Balance_End": "last",
          "Total_Property_Value": "last",
          "Total_Mortgage_Balance": "last",
          "Total_Property_Equity": "last",
          "Total_Property_Value_Ex_PPOR": "last",
          "Total_Mortgage_Balance_Ex_PPOR": "last",
          "Total_Property_Equity_Ex_PPOR": "last",
          "Net_Worth_Incl_PPOR": "last",
          "Net_Worth_Ex_PPOR": "last",
          "Target_Annual_Infl_Adj": "last",
          "Target_Monthly_Infl_Adj": "last",
          "Stock_Drawdown_Monthly": "last",
          "Super_Drawdown_Monthly": "last",
          "FIRE_Income_Monthly": "last",
          "Fire_Target_Monthly": "last",
          "Fire_Gap_Monthly": "last",
          "FIRE_Eligible": "last",
          "FIRE_Age": "last",

      }

      for p in property_list:
          prefix = p["name"].replace(" ", "_")
          agg.update({
              f"{prefix}_Mortgage_Balance_Start": "first",  # ✅ ADD THIS
              f"{prefix}_Mortgage_Balance": "last",
              f"{prefix}_Monthly_Mortgage_PMT": "sum",  # total paid that year
              f"{prefix}_Mortgage_Interest": "sum",
              f"{prefix}_Mortgage_Principal": "sum",
              f"{prefix}_Property_Value": "last",
              f"{prefix}_Rent_Income": "sum",
              f"{prefix}_Strata": "sum",
              f"{prefix}_Rates": "sum",
              f"{prefix}_Other_Costs": "sum",
              f"{prefix}_Offset_Allocated": "last",  # end-of-year allocated (snapshot)
              f"{prefix}_Net_Rent": "sum",
              f"{prefix}_Purchase_Cashflow": "sum",
              "Total_Property_Value": "last",
              "Total_Mortgage_Balance": "last",
              "Total_Property_Equity": "last",
              "Total_Property_Value_Ex_PPOR": "last",
              "Total_Mortgage_Balance_Ex_PPOR": "last",
              "Total_Property_Equity_Ex_PPOR": "last",
              "Net_Worth_Incl_PPOR": "last",
              "Net_Worth_Ex_PPOR": "last",
              "Target_Annual_Infl_Adj": "last",
              "Target_Monthly_Infl_Adj": "last",
          })

      dfy = dfm.groupby("Year", as_index=False).agg(agg)

      # Optional: your original "target income" (inflation-adjusted) yearly
      # dfy["Target_Income_Infl_Adj"] = inputs["todays_lifestyle_income"] * ((1 + inputs["inflation"]) ** (dfy["Year"] - start_year))

      # Make names closer to your old CSV (you can rename further)
      dfy = dfy.rename(columns={
          "Cash_Balance_End": "Cumulative_Savings",
          "Stock_Balance_End": "Total_Stock_Balance",
          "Tax_Paid": "Tax_Payable_Paid",
          "Total_Property_Value": "Property_Value_Total",
          "Total_Mortgage_Balance": "Mortgage_Balance_Total",
          "Total_Property_Equity": "Property_Equity_Total",
          "Salary_Paid": "Salary_Total",
          "Expenses_Paid": "Expenses_Total",
          "Stock_Contrib_Paid": "Stock_Contrib_Total",
          "Super_Balance_End": "Super_Balance",
          "Super_SG_Contribution": "Super_SG_Total",
          "Super_Extra_Contribution": "Super_Extra_Total",

      })

      

      # =========================
      # OUTPUT
      # =========================
      # Monthly detailed engine (debug)
      # print(dfm.to_csv(index=False))

      # Yearly summary (what you were printing before)
      dfy = dfy.round(0)
      print(dfy.to_csv(index=False))


    if display_month:
      dfm["Date"] = dfm["Date"].dt.strftime("%Y-%m-%d")
      out = dfm
      mode = "monthly"
    else:
        out = dfy
        mode = "yearly"
    
    rows = out.to_dict(orient="records")

    result = {
        "mode": mode,
        "columns": out.columns.tolist(),
        "rows": rows
    }

    return make_json_safe(result)


import math

def make_json_safe(x):
    """Convert NaN/Inf (and numpy scalars) into JSON-safe Python values."""
    try:
        import numpy as np
        numpy_int = (np.integer,)
        numpy_float = (np.floating,)
    except Exception:
        numpy_int = tuple()
        numpy_float = tuple()

    if isinstance(x, dict):
        return {k: make_json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_json_safe(v) for v in x]

    # numpy scalars -> python scalars
    if numpy_int and isinstance(x, numpy_int):
        return int(x)
    if numpy_float and isinstance(x, numpy_float):
        x = float(x)

    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None

    return x


result = run_fire_model(inputs_default, property_list_default, display_month=False)

# quick sanity checks
# print(result["mode"])
# print(len(result["rows"]), "rows")
# print(result["columns"][:20])
# print(result["rows"][0])   # first row
# print(result["rows"][-1])  # last row
