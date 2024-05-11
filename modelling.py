import pandas as pd
import numpy as np


def calc_cost_of_capital(interest_expense, pre_tax_cost_of_debt, average_maturity, bv_debt, num_shares_outstanding, curr_price, unlevered_beta, tax_rate, risk_free_rate, equity_risk_premium):
    market_value_of_debt = interest_expense * (1 - 1 / ((1 + pre_tax_cost_of_debt) ** average_maturity)) / pre_tax_cost_of_debt + bv_debt / ((1 + pre_tax_cost_of_debt) ** average_maturity)
    market_value_of_equity = num_shares_outstanding * curr_price
    market_value_of_capital = market_value_of_debt + market_value_of_equity
    equity_weight = market_value_of_equity / market_value_of_capital
    debt_weight = market_value_of_debt / market_value_of_capital

    levered_beta = unlevered_beta * (1 + (1 - tax_rate) * (market_value_of_debt / market_value_of_equity))

    cost_of_debt = pre_tax_cost_of_debt * (1 - tax_rate)
    cost_of_equity = risk_free_rate + levered_beta * equity_risk_premium

    cost_of_capital = cost_of_debt * debt_weight + cost_of_equity * equity_weight
    return cost_of_capital


def dcf(
    revenues,
    interest_expense,
    book_value_of_equity,
    book_value_of_debt,
    cash_and_marketable_securities,
    cross_holdings_and_other_non_operating_assets,
    number_of_shares_outstanding,
    curr_price,
    effective_tax_rate,
    marginal_tax_rate,
    unlevered_beta,
    risk_free_rate,
    equity_risk_premium,
    mature_erp,
    pre_tax_cost_of_debt,
    average_maturity,
    prob_of_failure,
    value_of_options,
    revenue_growth_rate_next_year,
    operating_margin_next_year,
    compounded_annual_revenue_growth_rate,
    target_pre_tax_operating_margin,
    year_of_convergence_for_margin,
    years_of_high_growth,
    sales_to_capital_ratio_early,
    sales_to_capital_ratio_steady,
):

    start_cost_of_capital = calc_cost_of_capital(interest_expense, pre_tax_cost_of_debt, average_maturity, book_value_of_debt, number_of_shares_outstanding, curr_price, unlevered_beta, marginal_tax_rate, risk_free_rate, equity_risk_premium)

    revenue_growth_rates = [0] + [revenue_growth_rate_next_year] + [compounded_annual_revenue_growth_rate] * (years_of_high_growth - 2) + np.linspace(compounded_annual_revenue_growth_rate, risk_free_rate, 6).tolist() + [risk_free_rate]
    df = pd.DataFrame({"revenue_growth_rate": revenue_growth_rates})
    df["revenues"] = revenues * (1 + df["revenue_growth_rate"]).cumprod()
    starting_operating_margin = operating_margin_next_year
    # TODO: Adjust operating income then get margin rather than make it an input
    df["operating_margin"] = (
        [starting_operating_margin] + np.linspace(operating_margin_next_year, target_pre_tax_operating_margin, year_of_convergence_for_margin).tolist() + [target_pre_tax_operating_margin] * (11 - year_of_convergence_for_margin)
    )
    df["operating_income"] = df["revenues"] * df["operating_margin"]
    df["tax_rate"] = [effective_tax_rate] * 6 + np.linspace(effective_tax_rate, marginal_tax_rate, 5).tolist() + [marginal_tax_rate]
    df["taxes"] = np.where(df["operating_income"] > 0, df["operating_income"] * df["tax_rate"], 0)

    df["nol"] = np.where(df["operating_income"] < 0, -df["operating_income"] * 0.8, 0)  # https://www.investopedia.com/terms/t/tax-loss-carryforward.asp
    df["nol_cumulative"] = df["nol"].cumsum()
    df["nol_utilized"] = np.where(df["operating_income"] > 0, np.minimum(df["nol_cumulative"], df["operating_income"]), 0)
    df["nol_cumulative"] -= df["nol_utilized"]
    df["taxes"] -= df["nol_utilized"] * df["tax_rate"]

    df["ebit_after_tax"] = df["operating_income"] - df["taxes"]
    df["reinvestment"] = (df["revenues"].diff(-1) * -1) / np.linspace(sales_to_capital_ratio_early, sales_to_capital_ratio_steady, 7).tolist() + [sales_to_capital_ratio_steady] * 5

    df.loc[0, "reinvestment"] = 0
    starting_invested_capital = book_value_of_equity + book_value_of_debt - cash_and_marketable_securities
    df["invested_capital"] = starting_invested_capital + df["reinvestment"].cumsum()
    df["roic"] = df["ebit_after_tax"] / df["invested_capital"]

    end_cost_of_capital = risk_free_rate + mature_erp
    df["cost_of_capital"] = [start_cost_of_capital] * 6 + np.linspace(start_cost_of_capital, end_cost_of_capital, 6).tolist()
    df.loc[11, "roic"] = end_cost_of_capital
    df.loc[11, "reinvestment"] = risk_free_rate / df.loc[11, "roic"] * df.loc[11, "ebit_after_tax"]

    df["fcff"] = df["ebit_after_tax"] - df["reinvestment"]
    df["discount_factor"] = (1 / (1 + df["cost_of_capital"])).cumprod()
    df["discount_factor"] = df["discount_factor"].shift(1)
    df["pv_fcff"] = df["fcff"] * df["discount_factor"]

    terminal_val = df.loc[11, "fcff"] / (end_cost_of_capital - risk_free_rate)
    terminal_pv = terminal_val * df.loc[10, "discount_factor"]
    pv_cf = df.loc[1:11, "pv_fcff"].sum() + terminal_pv

    proceeds_if_fail = pv_cf * 0.5
    op_value = pv_cf * (1 - prob_of_failure) + proceeds_if_fail * prob_of_failure

    value_of_equity = op_value - book_value_of_debt + cash_and_marketable_securities + cross_holdings_and_other_non_operating_assets
    value_of_equity = value_of_equity - value_of_options
    value_per_share = value_of_equity / number_of_shares_outstanding
    df.index = ["Base"] + list(range(1, 11)) + ["Terminal"]
    return value_per_share, df
