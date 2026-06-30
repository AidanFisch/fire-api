


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
    "stock_swr": 0.08,         # NEW: stock drawdown rate (annual)
    "super_swr": 0.04,         # NEW: super drawdown rate (annual)
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

def run_fire_model(inputs: dict | None, property_list: list | None, life_events: list | None = None, stock_contribution_overrides: list | None = None, display_month: bool = True):
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
    if life_events is None:
        life_events = []
    if stock_contribution_overrides is None:
        stock_contribution_overrides = []

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
    # CGT HELPERS
    # =========================

    def _parse_ym(ym: str):
        """'YYYY-MM' → (year, month) ints."""
        y, m = ym.split("-")
        return int(y), int(m)

    def _is_life_event_active(ev, yr, mo):
        """Is this income/expense-change event in effect for the given year/month?
        Open-ended (no 'end') means it runs for the rest of the model horizon."""
        start = ev.get("start")
        if not start:
            return False
        sy, sm = _parse_ym(start)
        cur_abs = yr * 12 + mo
        if cur_abs < sy * 12 + sm:
            return False
        end = ev.get("end")
        if end:
            ey, em = _parse_ym(end)
            if cur_abs > ey * 12 + em:
                return False
        return True

    def _is_ppor(prop, yr, mo):
        """Is this property the PPOR in the given year/month?"""
        periods = prop.get("ppor_periods")
        if not periods:
            return bool(prop.get("is_owner_occupied", False))
        for period in periods:
            fy, fm = _parse_ym(period["from"])
            if period.get("to"):
                ty, tm = _parse_ym(period["to"])
            else:
                ty, tm = 9999, 12
            after_start = (yr > fy) or (yr == fy and mo >= fm)
            before_end  = (yr < ty) or (yr == ty and mo <= tm)
            if after_start and before_end:
                return True
        return False

    def _marginal_rate(income):
        """Top marginal rate bracket (2025-26 schedule)."""
        if income <= 18200:   return 0.0
        elif income <= 45000: return 0.19
        elif income <= 120000: return 0.325
        elif income <= 180000: return 0.37
        else: return 0.45

    def _sale_year_month(p):
        """Return (year, month) for sale, using sale_date if present."""
        sd = p.get("sale_date")
        if sd and isinstance(sd, str) and len(sd) >= 7:
            try:
                y, m = sd[:7].split("-")
                return int(y), int(m)
            except Exception:
                pass
        yr = p.get("sale_year")
        if yr:
            return int(yr), 12   # legacy: assume December
        return None, None

    def _exempt_months(prop, buy_yr, sale_yr, buy_mo=1, sale_mo=12):
        """Total CGT-exempt months = genuine PPOR + 6-year-rule window."""
        p_abs = buy_yr * 12 + buy_mo   # actual purchase month
        s_abs = sale_yr * 12 + sale_mo  # exact sale month
        total  = s_abs - p_abs

        periods = prop.get("ppor_periods")
        if not periods:
            return total if prop.get("is_owner_occupied", False) else 0

        exempt = 0
        for period in periods:
            fy, fm = _parse_ym(period["from"])
            f_abs  = fy * 12 + fm
            if period.get("to"):
                ty, tm = _parse_ym(period["to"])
                t_abs  = ty * 12 + tm
            else:
                t_abs = s_abs

            # Genuine PPOR months within ownership window
            ps = max(f_abs, p_abs)
            pe = min(t_abs, s_abs)
            exempt += max(pe - ps, 0)

            # 6-year absence rule (only when period has an explicit end date)
            if period.get("six_year_rule") and period.get("to"):
                absence_start = t_abs
                # Find start of next PPOR period (caps the 6-year window early)
                next_ppor = None
                for other in periods:
                    ofy, ofm = _parse_ym(other["from"])
                    o_abs = ofy * 12 + ofm
                    if o_abs > t_abs and (next_ppor is None or o_abs < next_ppor):
                        next_ppor = o_abs
                six_yr_end = absence_start + 72   # 72 months = 6 years
                cap = min(six_yr_end,
                          next_ppor if next_ppor else s_abs,
                          s_abs)
                exempt += max(cap - absence_start, 0)

        return min(exempt, total)

    def _calc_cgt(prop, sale_yr, sale_price_val, annual_income,
                  cpi=0.025, no_30_min=False, sale_mo=12):
        """
        Return CGT tax payable on property sale.

        Applies:
          Bucket A (sell < Jul 2027):   50% discount + marginal rate
          Bucket B (buy < Jul 2027, sell >= Jul 2027): split at Jul 2027
          Bucket C (buy >= Jul 2027):   CPI indexation + 30% min tax
        PPOR & 6-year-rule exemption applied before bucket calc.
        """
        REFORM_ABS = 2027 * 12 + 7   # July 2027 (absolute month index)
        buy_yr, buy_mo = _buy_year_month(prop)
        cost_base = float(prop.get("purchase_price", 0)) + float(prop.get("purchase_fees", 0))

        total_gain = float(sale_price_val) - cost_base
        if total_gain <= 0:
            return 0.0

        buy_abs  = buy_yr * 12 + buy_mo
        sale_abs = sale_yr * 12 + sale_mo
        total_mo  = max(sale_abs - buy_abs, 1)
        held_long = (total_mo >= 12)

        exempt_mo  = _exempt_months(prop, buy_yr, sale_yr, buy_mo, sale_mo)
        tax_frac   = max(1.0 - exempt_mo / total_mo, 0.0)
        if tax_frac <= 0:
            return 0.0

        mr = _marginal_rate(float(annual_income))

        # ---- Bucket A: sell before 1 Jul 2027 ----
        if sale_abs < REFORM_ABS:
            gain = total_gain * tax_frac
            if held_long:
                gain *= 0.5          # 50% CGT discount
            return max(gain * mr, 0.0)

        # ---- Bucket C: bought on/after 1 Jul 2027 ----
        elif buy_abs >= REFORM_ABS:
            yrs = total_mo / 12.0
            indexed_cost = cost_base * ((1 + cpi) ** yrs)
            real_gain    = max(float(sale_price_val) - indexed_cost, 0.0) * tax_frac
            rate = mr if no_30_min else max(mr, 0.30)
            return max(real_gain * rate, 0.0)

        # ---- Bucket B: owned before Jul 2027, sell after ----
        else:
            pre_mo  = max(REFORM_ABS - buy_abs, 0)
            post_mo = max(sale_abs - REFORM_ABS, 0)
            pre_f   = pre_mo / total_mo
            post_f  = post_mo / total_mo

            # Pre-2027: old 50% discount
            pre_gain = total_gain * tax_frac * pre_f
            if held_long:
                pre_gain *= 0.5
            pre_tax = pre_gain * mr

            # Post-2027: CPI indexation on post period cost base
            post_sale_val  = float(sale_price_val) * tax_frac * post_f
            post_cost_base = cost_base * tax_frac * post_f * ((1 + cpi) ** (post_mo / 12.0))
            real_post_gain = max(post_sale_val - post_cost_base, 0.0)
            rate = mr if no_30_min else max(mr, 0.30)
            post_tax = real_post_gain * rate

            return max(pre_tax + post_tax, 0.0)

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

    def _buy_year_month(p) -> tuple[int, int]:
        """Return (year, month) for purchase, using purchase_date if present."""
        pd_str = p.get("purchase_date")
        if pd_str and isinstance(pd_str, str) and len(pd_str) >= 7:
            try:
                y, m = pd_str[:7].split("-")
                return int(y), int(m)
            except Exception:
                pass
        return int(p.get("year_bought", 2020)), 1

    def purchase_month(p) -> pd.Timestamp:
        y, m = _buy_year_month(p)
        return pd.Timestamp(f"{y}-{m:02d}-01")

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
        buy_y, buy_mo = _buy_year_month(p)
        # already owned if purchase is strictly before the model start month
        already_owned = (buy_y < start_year) or (buy_y == start_year and buy_mo == 1)
        if already_owned and not (buy_y == start_year and buy_mo > 1):
            prop_state[name]["active"] = True
            prop_state[name]["loan_balance"] = float(p["loan_balance_current"])
            prop_state[name]["property_value"] = float(p["current_value"])
            prop_state[name]["rent_monthly"] = float(p["monthly_rent"])

            # Elapsed months since purchase (using actual purchase month)
            elapsed = (start_year - buy_y) * 12 - buy_mo + 1
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
        dfm[f"{prefix}_Is_PPOR"]       = 0   # set dynamically each month in loop
        dfm[f"{prefix}_CGT_Paid"]      = 0.0
        dfm[f"{prefix}_Sale_Proceeds"] = 0.0
        dfm[f"{prefix}_Sale_Costs"]    = 0.0

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
    # ---- NEW: explicit withdrawals + passive income (monthly)
    dfm["Stock_Withdraw_Paid"] = 0.0
    dfm["Super_Withdraw_Paid"] = 0.0
    dfm["Withdrawals_Monthly"] = 0.0
    dfm["Passive_Income_Monthly"] = 0.0
    dfm["Cash_Drawdown_Paid"] = 0.0   # how much cash is used this month to cover deficit (bridge)


        # =========================
    # SPEED FIX #1 (arrays): cache reads + preallocate writes
    # =========================
    n = len(dfm)

    # Fast read arrays (things you read each month)
    dates = dfm["Date"].to_numpy()
    years_arr = dfm["Year"].to_numpy(dtype=np.int32)
    months_arr = dfm["Month"].to_numpy(dtype=np.int8)

    salary_monthly_arr = dfm["Salary_Monthly"].to_numpy(dtype=float)
    expenses_monthly_arr = dfm["Expenses_Monthly"].to_numpy(dtype=float)
    target_monthly_arr = (dfm["Target_Annual_Infl_Adj"].to_numpy(dtype=float) / 12.0)

    stock_contrib_monthly_arr = dfm["Stock_Contribution_Monthly"].to_numpy(dtype=float).copy()

    # Per-year ADDITIONAL stock contribution (e.g. "I put in an extra $5k in
    # 2027 from a bonus", or a negative amount because cash was tight that
    # year) — adds to the normal inflation-adjusted contribution for every
    # month in that calendar year, it does not replace it.
    if stock_contribution_overrides:
        additional_monthly_by_year = {
            int(o["year"]): float(o.get("amount", 0) or 0) / 12.0
            for o in stock_contribution_overrides if o.get("year") is not None
        }
        for _yr, _additional_monthly in additional_monthly_by_year.items():
            stock_contrib_monthly_arr[years_arr == _yr] += _additional_monthly
        dfm["Stock_Contribution_Monthly"] = stock_contrib_monthly_arr  # keep debug column in sync

    super_extra_monthly_arr = dfm["Super_Extra_Monthly"].to_numpy(dtype=float)
    age_arr = dfm["Age"].to_numpy(dtype=float)

    # Fast write arrays (global outputs you currently set with dfm.loc inside the loop)
    out_stock_growth = np.zeros(n, dtype=float)
    out_stock_end = np.zeros(n, dtype=float)

    out_super_growth = np.zeros(n, dtype=float)
    out_super_sg = np.zeros(n, dtype=float)
    out_super_extra = np.zeros(n, dtype=float)
    out_super_end = np.zeros(n, dtype=float)

    out_total_net_rent = np.zeros(n, dtype=float)
    out_tax_paid = np.zeros(n, dtype=float)
    out_cash_end = np.zeros(n, dtype=float)
    out_life_event_cashflow = np.zeros(n, dtype=float)

    out_stock_draw = np.zeros(n, dtype=float)
    out_super_draw = np.zeros(n, dtype=float)
    out_fire_income = np.zeros(n, dtype=float)

    out_fire_replacement = np.zeros(n, dtype=float)
    out_fire_principal_service = np.zeros(n, dtype=float)
    out_fire_target = np.zeros(n, dtype=float)
    out_fire_gap = np.zeros(n, dtype=float)

    out_fire_eligible = np.zeros(n, dtype=np.int8)
    out_fire_age = np.full(n, np.nan, dtype=float)

    out_salary_paid = np.zeros(n, dtype=float)
    out_stock_contrib_paid = np.zeros(n, dtype=float)

    out_stock_withdraw = np.zeros(n, dtype=float)
    out_super_withdraw = np.zeros(n, dtype=float)
    out_withdrawals = np.zeros(n, dtype=float)
    out_passive_income = np.zeros(n, dtype=float)
    out_cash_drawdown_paid = np.zeros(n, dtype=float)

    out_total_prop_value = np.zeros(n, dtype=float)
    out_total_mort_bal = np.zeros(n, dtype=float)

    out_net_worth_incl_ppor = np.zeros(n, dtype=float)
    out_net_worth_ex_ppor = np.zeros(n, dtype=float)
    out_net_worth_incl_super_incl_ppor = np.zeros(n, dtype=float)
    out_net_worth_incl_super_ex_ppor = np.zeros(n, dtype=float)

    out_total_prop_equity = np.zeros(n, dtype=float)
    out_total_prop_value_ex_ppor = np.zeros(n, dtype=float)
    out_total_mort_bal_ex_ppor = np.zeros(n, dtype=float)
    out_total_prop_equity_ex_ppor = np.zeros(n, dtype=float)

    out_offset_eligible_total = np.zeros(n, dtype=float)
    out_offset_alloc_total = np.zeros(n, dtype=float)
    out_offset_unused_cash = np.zeros(n, dtype=float)


    fired = False
    fire_age = None

    age_arr = dfm["Age"].to_numpy(dtype=float)

    for i in range(n):
        dt = dates[i]
        year = int(years_arr[i])
        month = int(months_arr[i])
        age_now = float(age_arr[i])


        # ---- Stock update (monthly)
        stock_growth_amt = stock_balance * stock_m_growth
        
        stock_contrib = 0.0 if fired else stock_contrib_monthly_arr[i]

        stock_balance += stock_growth_amt + stock_contrib


        out_stock_growth[i] = stock_growth_amt
        out_stock_end[i] = stock_balance


        # ---- Income/expenses (monthly)

        salary_m_raw = salary_monthly_arr[i]
        salary_m = salary_m_raw if not fired else 0.0

        expenses_m = expenses_monthly_arr[i]
        target_m = target_monthly_arr[i]

        # ---- Life events: temporary income/expense changes + one-off lump sums.
        # Applied before salary_m/expenses_m flow into super, tax, and cash below,
        # same way property cashflow is layered onto the base mortgage calc.
        life_event_cashflow_m = 0.0
        for ev in life_events:
            etype = ev.get("type")
            amt = float(ev.get("amount", 0) or 0)
            if etype == "income_change" and not fired and _is_life_event_active(ev, year, month):
                salary_m += salary_m * (amt / 100.0) if ev.get("mode") == "percent" else amt
            elif etype == "expense_change" and _is_life_event_active(ev, year, month):
                expenses_m += expenses_m * (amt / 100.0) if ev.get("mode") == "percent" else amt
            elif etype == "windfall" and ev.get("start"):
                ey, em = _parse_ym(ev["start"])
                if ey == year and em == month:
                    life_event_cashflow_m += amt
        salary_m = max(salary_m, 0.0)
        expenses_m = max(expenses_m, 0.0)

        stock_contrib_m = 0.0 if fired else stock_contrib



        out_salary_paid[i] = salary_m
        out_stock_contrib_paid[i] = stock_contrib


        # ---- Super update (monthly)
        # ---- Super update (monthly)
        super_sg_m = 0.0 if fired else (salary_m * float(inputs.get("super_sg_rate", 0.0)))  # ✅ SG stops too
        super_extra_m = float(super_extra_monthly_arr[i])
                        

        super_growth_amt = super_balance * super_m_growth
        super_balance += super_growth_amt + super_sg_m + super_extra_m

        out_super_growth[i] = super_growth_amt
        out_super_sg[i] = super_sg_m
        out_super_extra[i] = super_extra_m
        out_super_end[i] = super_balance


        # ---- Activate purchases (if year_bought == this month)
        purchase_cashflow_total = 0.0
        for p in property_list:
            prefix = p["name"].replace(" ", "_")
            buy_dt = purchase_month(p)

            if dt == buy_dt and (buy_dt.year > start_year or (buy_dt.year == start_year and buy_dt.month > 1)):
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

        eligible_total = float(sum(eligible_balances.values()))
        alloc_total = float(sum(offset_allocations.values()))

        out_offset_eligible_total[i] = eligible_total
        out_offset_alloc_total[i] = alloc_total
        out_offset_unused_cash[i] = max(available_offset_cash - alloc_total, 0.0)




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
            is_ppor_now = _is_ppor(p, year, month)
            dfm.loc[i, f"{prefix}_Is_PPOR"] = int(is_ppor_now)

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
                if not is_ppor_now:
                    prop_state[prefix]["rent_monthly"] *= monthly_growth_factor(p["rental_growth"])

                strata_m = (p["strata_quarterly"] * 4) / 12.0
                rates_m  = (p["rates_quarterly"] * 4) / 12.0
                other_m  = p["other_costs_monthly"]

                # paid off => no interest/principal/payment
                interest_m = 0.0
                principal_m = 0.0
                actual_pmt_m = 0.0
                offset_alloc = 0.0

                if is_ppor_now:
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

                # ---- Sale event (paid-off property) ----
                _sale_yr, _sale_mo = _sale_year_month(p)
                if _sale_yr and int(_sale_yr) == year and int(_sale_mo) == month:
                    _sp = p.get("sale_price")
                    _sale_px = float(_sp) if _sp else prop_state[prefix]["property_value"]
                    _cgt = _calc_cgt(p, int(_sale_yr), _sale_px, salary_m * 12,
                                     float(inputs.get("cpi_rate", 0.025)),
                                     bool(inputs.get("on_income_support", False)),
                                     sale_mo=int(_sale_mo))
                    _agent_rate   = float(p.get("sale_agent_rate", 0.022))
                    _conveyancing = float(p.get("sale_conveyancing", 2500))
                    _sale_costs   = _sale_px * _agent_rate + _conveyancing
                    _net_sale = _sale_px - _cgt - _sale_costs   # loan is 0
                    cash_balance += _net_sale
                    dfm.loc[i, f"{prefix}_CGT_Paid"]      = _cgt
                    dfm.loc[i, f"{prefix}_Sale_Costs"]    = _sale_costs
                    dfm.loc[i, f"{prefix}_Sale_Proceeds"]  = _net_sale
                    prop_state[prefix]["active"]           = False
                    prop_state[prefix]["property_value"]   = 0.0

                continue



            # Grow property value + rent monthly
            prop_state[prefix]["property_value"] *= monthly_growth_factor(p["property_growth"])
            if not is_ppor_now:
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
            if is_ppor_now:
                # PPOR: no rent, interest isn't tax-deductible (so it stays out of
                # net_rent_m / taxable income), but it's still real cash leaving your
                # pocket every month — must be counted in owner_cost_m or it vanishes
                # from the cash balance entirely.
                owner_cost_m = interest_m + strata_m + rates_m + other_m
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

            # ---- Sale event (property with loan) ----
            _sale_yr, _sale_mo = _sale_year_month(p)
            if _sale_yr and int(_sale_yr) == year and int(_sale_mo) == month:
                _sp = p.get("sale_price")
                _sale_px = float(_sp) if _sp else prop_state[prefix]["property_value"]
                _cgt = _calc_cgt(p, int(_sale_yr), _sale_px, salary_m * 12,
                                 float(inputs.get("cpi_rate", 0.025)),
                                 bool(inputs.get("on_income_support", False)),
                                 sale_mo=int(_sale_mo))
                _rem_loan     = float(max(prop_state[prefix]["loan_balance"], 0.0))
                _agent_rate   = float(p.get("sale_agent_rate", 0.022))
                _conveyancing = float(p.get("sale_conveyancing", 2500))
                _sale_costs   = _sale_px * _agent_rate + _conveyancing
                _net_sale = _sale_px - _rem_loan - _cgt - _sale_costs
                cash_balance += _net_sale
                dfm.loc[i, f"{prefix}_CGT_Paid"]      = _cgt
                dfm.loc[i, f"{prefix}_Sale_Costs"]    = _sale_costs
                dfm.loc[i, f"{prefix}_Sale_Proceeds"]  = _net_sale
                prop_state[prefix]["active"]           = False
                prop_state[prefix]["loan_balance"]     = 0.0
                prop_state[prefix]["property_value"]   = 0.0

        out_total_net_rent[i] = total_net_rent


        # =========================
        # FIRE VIABILITY CHECK (CASHFLOW-BASED)
        # =========================



        # Allowed deterministic drawdowns (DO NOT mutate balances here)
        stock_draw_m = (stock_balance * inputs["stock_swr"]) / 12.0

        super_draw_m = 0.0
        if age_now >= inputs["super_access_age"]:

            super_draw_m = (super_balance * inputs["super_swr"]) / 12.0
            

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

        target_m = target_monthly_arr[i]



        fire_gap_m = fire_replacement_cashflow_m - target_m

        out_stock_draw[i] = stock_draw_m
        out_super_draw[i] = super_draw_m
        out_fire_income[i] = fire_replacement_cashflow_m

        out_fire_replacement[i] = fire_replacement_cashflow_m
        out_fire_principal_service[i] = principal_service_m
        out_fire_target[i] = target_m
        out_fire_gap[i] = fire_gap_m

        eligible_now = fire_gap_m >= 0

        if (not fired) and eligible_now:
            fired = True
            fire_age = float(age_arr[i])


        out_fire_eligible[i] = 1 if fired else 0
        out_fire_age[i] = fire_age if fire_age is not None else np.nan


        # =========================
        # CASH DRAWDOWN (REPORTING ONLY)
        # Only after FIRE is achieved, and only until super access age.
        # This does NOT change cash_balance.
        # =========================
        # Cash drawdown needed to "maintain" the FIRE target gap (reporting-only)
        cash_draw_m = 0.0
        if fired:
            required_gap = max(-fire_gap_m, 0.0)          # how much extra cash is needed this month
            available_cash = max(cash_balance, 0.0)       # can't draw negative cash
            cash_draw_m = min(required_gap, available_cash)

        out_cash_drawdown_paid[i] = cash_draw_m




        # =========================
        # ACTUAL ASSET WITHDRAWALS (POST-FIRE ONLY)
        # =========================

        stock_withdraw_m = 0.0
        super_withdraw_m = 0.0

        if fired:
            stock_withdraw_m = min(stock_draw_m, stock_balance)
            stock_balance -= stock_withdraw_m

            if age_now >= inputs["super_access_age"]:
                super_withdraw_m = super_draw_m
                super_balance -= super_withdraw_m
                
        
        
        out_stock_withdraw[i] = stock_withdraw_m
        out_super_withdraw[i] = super_withdraw_m
        out_withdrawals[i] = stock_withdraw_m + super_withdraw_m
        out_passive_income[i] = total_net_rent + stock_withdraw_m + super_withdraw_m

        




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

            if _is_ppor(p, year, month):
                ppor_value += v
                ppor_mort += b

        # ✅ Write totals
        out_total_prop_value[i] = total_prop_value
        out_total_mort_bal[i] = total_mort_bal

        out_total_prop_equity[i] = max(total_prop_value - total_mort_bal, 0.0)

        out_total_prop_value_ex_ppor[i] = max(total_prop_value - ppor_value, 0.0)
        out_total_mort_bal_ex_ppor[i] = max(total_mort_bal - ppor_mort, 0.0)
        out_total_prop_equity_ex_ppor[i] = max(out_total_prop_value_ex_ppor[i] - out_total_mort_bal_ex_ppor[i], 0.0)



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
        out_tax_paid[i] = tax_paid_this_month







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
            + total_net_rent
            + stock_withdraw_m
            + super_withdraw_m
            + life_event_cashflow_m
            - expenses_m
            - stock_contrib_m
            - super_extra_m
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


        out_cash_end[i] = cash_balance
        out_life_event_cashflow[i] = life_event_cashflow_m

        out_net_worth_incl_ppor[i] = cash_balance + stock_balance + total_prop_value - total_mort_bal
        out_net_worth_ex_ppor[i] = (
            cash_balance
            + stock_balance
            + (total_prop_value - ppor_value)
            - (total_mort_bal - ppor_mort)
        )

        out_net_worth_incl_super_incl_ppor[i] = cash_balance + stock_balance + super_balance + total_prop_value - total_mort_bal
        out_net_worth_incl_super_ex_ppor[i] = (
            cash_balance
            + stock_balance
            + super_balance
            + (total_prop_value - ppor_value)
            - (total_mort_bal - ppor_mort)
        )

    # =========================
    # SPEED FIX #1 (arrays): write arrays back to dfm
    # =========================
    dfm["Stock_Growth_Accrued"] = out_stock_growth
    dfm["Stock_Balance_End"] = out_stock_end

    dfm["Super_Growth_Accrued"] = out_super_growth
    dfm["Super_SG_Contribution"] = out_super_sg
    dfm["Super_Extra_Contribution"] = out_super_extra
    dfm["Super_Balance_End"] = out_super_end

    dfm["Total_Net_Rent"] = out_total_net_rent
    dfm["Tax_Paid"] = out_tax_paid
    dfm["Cash_Balance_End"] = out_cash_end
    dfm["Life_Event_Cashflow"] = out_life_event_cashflow

    dfm["Stock_Drawdown_Monthly"] = out_stock_draw
    dfm["Super_Drawdown_Monthly"] = out_super_draw
    dfm["FIRE_Income_Monthly"] = out_fire_income

    dfm["Fire_Replacement_Cashflow_Monthly"] = out_fire_replacement
    dfm["Fire_Principal_Service_Monthly"] = out_fire_principal_service
    dfm["Fire_Target_Monthly"] = out_fire_target
    dfm["Fire_Gap_Monthly"] = out_fire_gap

    dfm["FIRE_Eligible"] = out_fire_eligible
    dfm["FIRE_Age"] = out_fire_age

    dfm["Salary_Paid"] = out_salary_paid
    dfm["Stock_Contrib_Paid"] = out_stock_contrib_paid

    dfm["Stock_Withdraw_Paid"] = out_stock_withdraw
    dfm["Super_Withdraw_Paid"] = out_super_withdraw
    dfm["Withdrawals_Monthly"] = out_withdrawals
    dfm["Passive_Income_Monthly"] = out_passive_income
    dfm["Cash_Drawdown_Paid"] = out_cash_drawdown_paid

    dfm["Total_Property_Value"] = out_total_prop_value
    dfm["Total_Mortgage_Balance"] = out_total_mort_bal

    dfm["Net_Worth_Incl_PPOR"] = out_net_worth_incl_ppor
    dfm["Net_Worth_Ex_PPOR"] = out_net_worth_ex_ppor
    dfm["Net_Worth_Incl_Super_Incl_PPOR"] = out_net_worth_incl_super_incl_ppor
    dfm["Net_Worth_Incl_Super_Ex_PPOR"] = out_net_worth_incl_super_ex_ppor

    dfm["Total_Property_Equity"] = out_total_prop_equity
    dfm["Total_Property_Value_Ex_PPOR"] = out_total_prop_value_ex_ppor
    dfm["Total_Mortgage_Balance_Ex_PPOR"] = out_total_mort_bal_ex_ppor
    dfm["Total_Property_Equity_Ex_PPOR"] = out_total_prop_equity_ex_ppor

    dfm["Offset_Eligible_Balance_Total"] = out_offset_eligible_total
    dfm["Offset_Allocated_Total"] = out_offset_alloc_total
    dfm["Offset_Unused_Cash"] = out_offset_unused_cash


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
          "Life_Event_Cashflow": "sum",
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
          "Stock_Drawdown_Monthly": "last",
          "Super_Drawdown_Monthly": "last",
          "FIRE_Income_Monthly": "last",
          "Fire_Target_Monthly": "last",
          "Fire_Gap_Monthly": "last",
          "FIRE_Eligible": "last",
          "FIRE_Age": "last",
          
          "Stock_Withdraw_Paid": "sum",
          "Super_Withdraw_Paid": "sum",
          "Withdrawals_Monthly": "sum",
          "Passive_Income_Monthly": "sum",
          "Cash_Drawdown_Paid": "sum",
          "Cash_Drawdown_Monthly_Last": "last",


      }

      

      for p in property_list:
          prefix = p["name"].replace(" ", "_")
          agg.update({
              f"{prefix}_Mortgage_Balance_Start": "first",
              f"{prefix}_Mortgage_Balance": "last",
              f"{prefix}_Monthly_Mortgage_PMT": "sum",
              f"{prefix}_Mortgage_Interest": "sum",
              f"{prefix}_Mortgage_Principal": "sum",
              f"{prefix}_Property_Value": "last",
              f"{prefix}_Rent_Income": "sum",
              f"{prefix}_Strata": "sum",
              f"{prefix}_Rates": "sum",
              f"{prefix}_Other_Costs": "sum",
              f"{prefix}_Offset_Allocated": "last",
              f"{prefix}_Net_Rent": "sum",
              f"{prefix}_Purchase_Cashflow": "sum",
              f"{prefix}_CGT_Paid": "sum",
              f"{prefix}_Sale_Costs": "sum",
              f"{prefix}_Sale_Proceeds": "sum",
              f"{prefix}_Is_PPOR": "last",
              "Total_Property_Value": "last",
              "Total_Mortgage_Balance": "last",
              "Total_Property_Equity": "last",
              "Total_Property_Value_Ex_PPOR": "last",
              "Total_Mortgage_Balance_Ex_PPOR": "last",
              "Total_Property_Equity_Ex_PPOR": "last",
              "Net_Worth_Incl_PPOR": "last",
              "Net_Worth_Ex_PPOR": "last",
              "Target_Annual_Infl_Adj": "last",
          })

      dfm["Cash_Drawdown_Monthly_Last"] = dfm["Cash_Drawdown_Paid"]

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
          "Stock_Withdraw_Paid": "Stock_Withdraw_Total",
          "Super_Withdraw_Paid": "Super_Withdraw_Total",
          "Withdrawals_Monthly": "Withdrawals_Total",
          "Passive_Income_Monthly": "Passive_Income_Total",
          "Cash_Drawdown_Paid": "Cash_Drawdown_Total",
          "Cash_Drawdown_Monthly_Last": "Cash_Drawdown_Monthly",

      })



      # =========================
      # OUTPUT
      # =========================
      # Monthly detailed engine (debug)
      # print(dfm.to_csv(index=False))

      # Yearly summary (what you were printing before)
      dfy = dfy.round(0)


    if display_month:
      dfm["Date"] = dfm["Date"].dt.strftime("%Y-%m-%d")
      out = dfm
      mode = "monthly"
    else:
        out = dfy
        mode = "yearly"

    out = out.replace({np.nan: None})
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
