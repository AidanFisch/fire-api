

  
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

property_2_input = {
    "id": 2,  
    "name": "House",
    "purchase_price": 750000,
    "current_value": 820000,
    "original_loan": 600000,
    "loan_balance_current": 540000,
    "purchase_fees": 35000,
    "monthly_rent": 3200,
    "strata_quarterly": 1200,
    "rates_quarterly": 900,
    "other_costs_monthly": 200,
    "interest_rate": 0.055,
    "property_growth": 0.03,
    "rental_growth": 0.03,
    "year_bought": 2028,
    "loan_term_years": 30,
    "use_offset": True,
    "is_owner_occupied": False

}

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

property_list_default = [property_1_input, property_2_input, property_3_input]
# display_month = False




from typing import Optional

def run_fire_model(inputs: dict | None, property_list: list | None, display_month: bool = True):
    import pandas as pd
    import numpy as np
    from datetime import date



    # Fallbacks for notebook testing
    if inputs is None:
        inputs = inputs_default
    if property_list is None:
        property_list = property_list_default

    ###debugging###
    required_input_keys = set(inputs_default.keys())
    missing = required_input_keys - set(inputs.keys())
    if missing:
        raise ValueError(f"Missing inputs keys: {sorted(missing)}")

    # validate properties have required keys too
    required_prop_keys = set().union(*(p.keys() for p in property_list_default))

    for idx, p in enumerate(property_list):
        missing_p = required_prop_keys - set(p.keys())
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
    end_year = start_year + (inputs["end_age"] - inputs["current_age"])
    months = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq="MS")  # month-start

    # Build base monthly dataframe
    dfm = pd.DataFrame({"Date": months})
    dfm["Year"] = dfm["Date"].dt.year
    dfm["Month"] = dfm["Date"].dt.month
    dfm["t_months"] = range(len(dfm))  # 0..N
    dfm["Age"] = inputs["current_age"] + dfm["t_months"] / 12.0

    # Salary/expenses grow with YEARS FROM NOW (not "i from birth")
    dfm["t_years"] = dfm["t_months"] / 12.0

    # Use annual compounding continuously via (1+g)^(t_years)
    dfm["Salary_Annual"] = inputs["current_income"] * ((1 + inputs["average_salary_increase"]) ** dfm["t_years"])
    dfm["Expenses_Annual"] = inputs["current_expenses"] * ((1 + inputs["inflation"]) ** dfm["t_years"])

    dfm["Salary_Monthly"] = dfm["Salary_Annual"] / 12.0
    dfm["Expenses_Monthly"] = dfm["Expenses_Annual"] / 12.0

    # Stocks monthly
    stock_m_growth = monthly_growth_factor(inputs["stock_growth"]) - 1
    dfm["Stock_Contribution_Annual"] = inputs["stock_yearly_contribution"] * ((1 + inputs["inflation"]) ** dfm["t_years"])
    dfm["Stock_Contribution_Monthly"] = dfm["Stock_Contribution_Annual"] / 12.0



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



    # Create columns per property
    for p in property_list:
        prefix = p["name"].replace(" ", "_")
        for col in [
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

    for p in property_list:
      prefix = p["name"].replace(" ", "_")
      dfm[f"{prefix}_Is_PPOR"] = int(p.get("is_owner_occupied", False))

    dfm["Total_Net_Rent"] = 0.0
    dfm["Tax_Paid"] = 0.0
    dfm["Cash_Balance_End"] = 0.0
    dfm["Stock_Balance_End"] = 0.0
    dfm["Stock_Growth_Monthly"] = 0.0
    dfm["Offset_Eligible_Balance_Total"] = 0.0
    dfm["Offset_Allocated_Total"] = 0.0
    dfm["Offset_Unused_Cash"] = 0.0

    for i, row in dfm.iterrows():
        dt = row["Date"]
        year = int(row["Year"])
        month = int(row["Month"])


        # ---- Stock update (monthly)
        stock_growth_amt = stock_balance * stock_m_growth
        stock_balance += stock_growth_amt + row["Stock_Contribution_Monthly"]

        dfm.loc[i, "Stock_Growth_Monthly"] = stock_growth_amt
        dfm.loc[i, "Stock_Balance_End"] = stock_balance

        # ---- Income/expenses (monthly)
        salary_m = row["Salary_Monthly"]
        expenses_m = row["Expenses_Monthly"]
        stock_contrib_m = row["Stock_Contribution_Monthly"]

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


        # ---- Offset allocation base: only positive cash can offset
        # Determine total active mortgage balance for pro-rata allocation
    # ---- Offset allocation base (WEIGHTED across OFFSET-ENABLED loans only)
        available_offset_cash = max(cash_balance, 0.0)

        eligible_balances = {}
        for p in property_list:
            prefix = p["name"].replace(" ", "_")
            if prop_state[prefix]["active"] and p.get("use_offset", False):
                bal = float(max(prop_state[prefix]["loan_balance"], 0.0))
                if bal > 0:
                    eligible_balances[prefix] = bal

        # Allocate the cash pool across eligible loans, with caps + redistribution
        offset_allocations = allocate_weighted_offset(available_offset_cash, eligible_balances)
        eligible_total = sum(eligible_balances.values())
        allocated_total = sum(offset_allocations.values())

        dfm.loc[i, "Offset_Eligible_Balance_Total"] = eligible_total
        dfm.loc[i, "Offset_Allocated_Total"] = allocated_total
        dfm.loc[i, "Offset_Unused_Cash"] = max(available_offset_cash - allocated_total, 0.0)




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
                continue

            loan_bal = prop_state[prefix]["loan_balance"]
            if loan_bal <= 0:
                # paid off
                prop_state[prefix]["loan_balance"] = 0.0
                dfm.loc[i, f"{prefix}_Mortgage_Balance"] = 0.0
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

            



            # Mortgage calc (monthly)
            r_m = monthly_rate(p["interest_rate"])
            interest_m = max(0.0, loan_bal - offset_alloc) * r_m
            pmt_m = prop_state[prefix]["monthly_pmt"]

            # Final-month guard so balance never goes negative
            principal_m = min(max(pmt_m - interest_m, 0.0), loan_bal)
            # If the scheduled PMT is larger than needed at the end, true payment is interest+principal
            actual_pmt_m = interest_m + principal_m

            loan_bal_end = loan_bal - principal_m
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

        # ---- PAYG SMOOTHED TAX (monthly withholding)

        # Annualise current run-rate income
        annualised_taxable_income = (salary_m + total_net_rent) * 12.0

        # Estimate full-year tax
        estimated_annual_tax = tax_payable(annualised_taxable_income)

        # Pay 1/12 each month
        tax_paid_this_month = estimated_annual_tax / 12.0

        dfm.loc[i, "Tax_Paid"] = tax_paid_this_month

        # ---- Taxable income accrues monthly (salary + net rent). Tax paid once at December.


        dfm.loc[i, "Tax_Paid"] = tax_paid_this_month

        # ---- Cash balance update (THIS is your offset pool)
        # Cashflow rule:
        # cash += salary - expenses - stock_contrib + net_rent - principal - tax + purchase_cashflow
        # (interest is already subtracted inside net_rent, so we do NOT subtract interest again)
        cash_balance = (
            cash_balance
            + salary_m
            - expenses_m
            - stock_contrib_m
            + total_net_rent
            - total_owner_costs 
            - total_principal
            - tax_paid_this_month
        )

        dfm.loc[i, "Cash_Balance_End"] = cash_balance


    ### toggle display
    
    if display_month == True:

      # Round numeric columns only
      # Round money columns only (keep time columns accurate)
      do_not_round = {"t_months", "t_years", "Age", "Year", "Month"}
      numeric_cols = [c for c in dfm.select_dtypes(include=["number"]).columns if c not in do_not_round]
      dfm[numeric_cols] = dfm[numeric_cols].round(0)

      dfm["Age"] = dfm["Age"].round(2)
      dfm["t_years"] = dfm["t_years"].round(3)

      print(dfm.to_csv(index=False))


    else:
      # =========================
      # ROLL UP TO YEARLY REPORT (end-of-year)
      # =========================

      # Define aggregation: sums for flows, last for balances/values
      agg = {
          "Age": "last",
          "Salary_Annual": "last",           # end-of-year salary run rate
          "Expenses_Annual": "last",         # end-of-year expense run rate
          "Salary_Monthly": "last",
          "Expenses_Monthly": "last",
          "Stock_Contribution_Annual": "last",
          "Stock_Contribution_Monthly": "sum",
          "Stock_Growth_Monthly": "sum",
          "Stock_Balance_End": "last",
          "Total_Net_Rent": "sum",
          "Tax_Paid": "sum",
          "Cash_Balance_End": "last",
      }

      for p in property_list:
          prefix = p["name"].replace(" ", "_")
          agg.update({
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
          })

      dfy = dfm.groupby("Year", as_index=False).agg(agg)

      # Optional: your original "target income" (inflation-adjusted) yearly
      dfy["Target_Income_Infl_Adj"] = inputs["todays_lifestyle_income"] * ((1 + inputs["inflation"]) ** (dfy["Year"] - start_year))

      # Make names closer to your old CSV (you can rename further)
      dfy = dfy.rename(columns={
          "Cash_Balance_End": "Cumulative_Savings",
          "Stock_Balance_End": "Total_Stock_Balance",
          "Tax_Paid": "Tax_Payable_Paid",
      })

      # =========================
      # OUTPUT
      # =========================
      # Monthly detailed engine (debug)
      # print(dfm.to_csv(index=False))

      # Yearly summary (what you were printing before)
      dfy = dfy.round(0).astype(int)
      print(dfy.to_csv(index=False))


    if display_month:
      dfm["Date"] = dfm["Date"].dt.strftime("%Y-%m-%d")
      out = dfm
      mode = "monthly"
    else:
        out = dfy
        mode = "yearly"

    return {
        "mode": mode,
        "columns": out.columns.tolist(),
        "rows": out.to_dict(orient="records")
    }

result = run_fire_model(inputs_default, property_list_default, display_month=True)

# # quick sanity checks
# print(result["mode"])
# print(len(result["rows"]), "rows")
# print(result["columns"][:20])
# print(result["rows"][0])   # first row
# print(result["rows"][-1])  # last row
