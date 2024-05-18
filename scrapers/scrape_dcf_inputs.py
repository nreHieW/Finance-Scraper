import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
from bs4 import BeautifulSoup
import warnings
import datetime
from sentence_transformers import SentenceTransformer
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import json
import concurrent.futures
import time

load_dotenv()

warnings.filterwarnings("ignore")

headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36"}

MAX_WORKERS = 120


# https://stackoverflow.com/questions/30098263/inserting-a-document-with-pymongo-invaliddocument-cannot-encode-object
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(CustomEncoder, self).default(obj)


class StringMapper:
    def __init__(self, gts: list, threshold=0.85):
        self.model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
        self.gts = gts
        self.embeddings = self.model.encode(gts)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1)[:, None]
        self.threshold = threshold

    def get_closest(self, query: str, num_results=3):
        if query in self.gts or query.lower() in self.gts:
            return [query]

        query_words = set(query.lower().split())
        candidates = []
        for gt in self.gts:
            gt_words = set(gt.lower().split())
            if any(len(word) >= 3 for word in query_words & gt_words):
                candidates.append(gt)
        if candidates:
            candidates = sorted(candidates, key=lambda x: len(x.split()))
            return [candidates[0]]

        query_embedding = self.model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(self.embeddings, query_embedding)
        similarities = np.argsort(similarities)[::-1]
        similarities = similarities[similarities > self.threshold][:num_results]
        if len(similarities) == 0:
            return None
        return [self.gts[i] for i in similarities]

    def get_closest_with_scores(self, query: str, num_results=3, indices_to_adjust=None):
        if query in self.gts or query.lower() in self.gts:
            return [(query, 1.0)]

        query_words = set(query.lower().split())
        candidates = []
        for gt in self.gts:
            gt_words = set(gt.lower().split())
            if any(len(word) >= 3 for word in query_words & gt_words):
                candidates.append(gt)
        if candidates:
            candidates = sorted(candidates, key=lambda x: len(x.split()))
            return [(candidates[0], 1.0)]

        query_embedding = self.model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        scores = np.dot(self.embeddings, query_embedding)
        if indices_to_adjust:
            scores[indices_to_adjust] += np.max(scores) * 0.1

        similarities = np.argsort(scores)[::-1]
        similarities = similarities[similarities > self.threshold][:num_results]
        if len(similarities) == 0:
            return None
        return [(self.gts[i], scores[i]) for i in similarities]


def get_regional_crps(revenues_by_region: dict, mapper: StringMapper, country_erps: dict):
    regions = list(revenues_by_region.keys())
    indices_to_adjust = [i for i in range(len(mapper.gts) - 10, len(mapper.gts))]
    mappings = [mapper.get_closest_with_scores(x, indices_to_adjust=indices_to_adjust) for x in regions]

    flattened_mappings = [(region, gt, score) for region, mapping in zip(regions, mappings) if mapping for gt, score in mapping]
    flattened_mappings.sort(key=lambda x: x[2], reverse=True)
    used_gts = set()
    final_mappings = {}

    for region, gt, score in flattened_mappings:
        if gt not in used_gts:
            final_mappings[region] = gt
            used_gts.add(gt)
    for region in regions:
        if region not in final_mappings:
            final_mappings[region] = "Global"
    # print(mapper.gts)
    crps = [country_erps[final_mappings[region]] for region in regions]
    total_revenues = sum(revenues_by_region.values())
    weights = [revenues_by_region[region] / total_revenues for region in regions]
    # print(final_mappings)
    return sum([x * y for x, y in zip(crps, weights)]), {final_mappings[region]: v for region, v in revenues_by_region.items()}


def get_industry_beta(industry: str, mapper: StringMapper, industry_betas: dict):
    industry = mapper.get_closest(industry)[0]
    if industry is None:
        industry = "Grand Total"
    return industry_betas[industry], industry


def get_10year_tbill():
    url = "https://www.cnbc.com/quotes/US10Y"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    res = soup.find_all("span", class_="QuoteStrip-lastPrice")[0].text.replace("%", "")
    return float(res) / 100


def get_mature_erp():
    url = "https://pages.stern.nyu.edu/~adamodar/pc/implprem/ERPbymonth.xlsx"
    page = requests.get(url, verify=False)
    return pd.read_excel(page.content)["ERP (T12m)"].iloc[-1]


def synthetic_rating(market_cap, operating_income, interest_expense):
    if market_cap > 5 * 1e9:
        rating_mapping = [
            [-100000.0, 0.199999, "D", "20.00%"],
            [0.2, 0.649999, "C", "17.00%"],
            [0.65, 0.799999, "CC", "11.78%"],
            [0.8, 1.249999, "CCC", "8.51%"],
            [1.25, 1.499999, "B-", "5.24%"],
            [1.5, 1.749999, "B", "3.61%"],
            [1.75, 1.999999, "B+", "3.14%"],
            [2.0, 2.2499999, "BB", "2.21%"],
            [2.25, 2.49999, "BB+", "1.74%"],
            [2.5, 2.999999, "BBB", "1.47%"],
            [3.0, 4.249999, "A-", "1.21%"],
            [4.25, 5.499999, "A", "1.07%"],
            [5.5, 6.499999, "A+", "0.92%"],
            [6.5, 8.499999, "AA", "0.70%"],
            [8.5, 100000.0, "AAA", "0.59%"],
        ]
    else:
        rating_mapping = [
            [0.5, 0.799999, "C", "17.00%"],
            [0.8, 1.249999, "CC", "11.78%"],
            [1.25, 1.499999, "CCC", "8.51%"],
            [1.5, 1.999999, "B-", "5.24%"],
            [2.0, 2.499999, "B", "3.61%"],
            [2.5, 2.999999, "B+", "3.14%"],
            [3.0, 3.499999, "BB", "2.21%"],
            [3.5, 3.9999999, "BB+", "1.74%"],
            [4.0, 4.499999, "BBB", "1.47%"],
            [4.5, 5.999999, "A-", "1.21%"],
            [6.0, 7.499999, "A", "1.07%"],
            [7.5, 9.499999, "A+", "0.92%"],
            [9.5, 12.499999, "AA", "0.70%"],
            [12.5, 100000.0, "AAA", "0.59%"],
        ]

    if interest_expense <= 0:
        interest_coverage_rato = 100000
    elif operating_income <= 0:
        interest_coverage_rato = -100000
    else:
        interest_coverage_rato = operating_income / interest_expense

    rating, spread = None, None
    for low, high, r, s in rating_mapping:
        if low <= interest_coverage_rato <= high:
            rating, spread = r, s
            break
    default_prob = {"AAA": 0.70, "AA": 0.72, "A+": 0.72, "A": 1.24, "A-": 1.24, "BBB": 3.32, "BB+": 3.32, "BB": 11.78, "B+": 11.79, "B": 23.74, "B-": 23.75, "CCC": 50.38, "CC": 50.38, "C": 50.38, "D": 50.38}[rating] / 100  # in percentages
    spread = float(s.replace("%", "")) / 100
    return rating, spread, default_prob


def get_country_erp():
    url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx"
    page = requests.get(url, verify=False)
    country_premium = pd.read_excel(page.content, sheet_name="ERPs by country", skiprows=6)
    countries = country_premium.iloc[:156][["Country", "Country Risk Premium"]].set_index("Country").to_dict()["Country Risk Premium"]
    frontier_countries = country_premium.iloc[158:179][["Country", "Moody's rating"]].set_index("Country").to_dict()["Moody's rating"]  # Table is concat at the bottom
    regions = pd.read_excel(page.content, sheet_name="Regional Weighted Averages")
    regions = regions.iloc[169:179][["Country", "Moody's rating"]].set_index("Country").to_dict()["Moody's rating"]
    return {**countries, **frontier_countries, **regions}


def get_industry_avgs():
    url = "https://pages.stern.nyu.edu/~adamodar/pc/fcffsimpleginzu.xlsx"
    page = requests.get(url, verify=False)
    df = pd.read_excel(page.content, sheet_name="Industry Averages(US)")
    return df.set_index("Industry Name").to_dict()


def get_marketscreener_url(ticker):
    search_url = "https://www.marketscreener.com/search/?q=" + "+".join(ticker.split())
    page = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(page.content, "lxml")
    rows = soup.find_all("tr")
    found_link = None
    for row in rows:
        currency_tag = row.find("span", {"class": "txt-muted"})
        if currency_tag:
            currency = currency_tag.text.strip()
            if currency == "USD":
                link = row.find("a", href=True)["href"]
                found_link = "https://www.marketscreener.com" + link
                break
    if not found_link:
        print(f"[INFO] Could not find {ticker} on marketscreener")
    return found_link


def get_revenue_by_region(ticker, url):
    page = requests.get(url + "company/", headers=headers)
    soup = BeautifulSoup(page.content, "lxml")
    df = None
    for div in soup.find_all("div", {"class": "card mb-15 card--collapsible card--scrollable"}):
        header_text = div.find("div", {"class": "card-header"}).text
        if header_text == "Sales per region":
            df = pd.read_html(str(div.find("table")))[0]
            break
    if df is None:
        print(f"[INFO] Could not find sales per region for {ticker}")
    countries = df[df.columns[0]].values
    countries = [re.search(r"^([^\d]+)", item).group(0).strip() for item in countries]
    df["country"] = countries
    df.set_index("country", inplace=True)
    numeric_col_names = [x for x in df.columns if x.isdigit()]
    latest_year = max([int(x) for x in numeric_col_names])
    df = df[numeric_col_names]
    return df[str(latest_year)].to_dict()


def get_revenue_forecasts(ticker, url):
    page = requests.get(url + "finances/", headers=headers)
    soup = BeautifulSoup(page.content, features="lxml")
    for div in soup.find_all("div", {"class": "card card--collapsible mb-15"}):
        header_text = div.find("div", {"class": "card-header"}).text.lower()
        if "income statement" in header_text and "annual" in header_text:
            income_statement = pd.read_html(str(div.find("table")))[0]
            income_statement = income_statement.dropna(axis=1, how="all")
            income_statement.iloc[:, 0] = income_statement.iloc[:, 0].str.replace(r"\d", "", regex=True)
            income_statement.set_index(income_statement.columns[0], inplace=True)
            income_statement.index = income_statement.index.str.strip()

            indiv = income_statement.loc[["Net sales"]]
            curr_year = datetime.datetime.now().year - 1
            curr_year_index = indiv.columns.get_loc(str(curr_year))
            indiv = indiv.iloc[:, curr_year_index:].astype(float)
            growth = indiv.pct_change(axis=1)
            return growth.values[0][1], growth.values[0, 1:].mean()


def get_dcf_inputs(ticker: str, country_erps: dict, region_mapper: StringMapper, avg_metrics: dict, industry_mapper: StringMapper, mature_erp: float, risk_free_rate: float):
    # Defaults
    average_maturity = 0
    marginal_tax_rate = 0.21
    value_of_options = 0

    ticker = yf.Ticker(ticker)
    q_income_statement = ticker.quarterly_income_stmt
    ttm_income_statement = q_income_statement[q_income_statement.columns[:4]].T
    last_balance_sheet = ticker.quarterly_balance_sheet
    last_balance_sheet = last_balance_sheet[last_balance_sheet.columns[:4]].T
    info = ticker.info
    name = info.get("longName")

    revenues = ttm_income_statement.get("Operating Revenue", pd.Series([0] * len(ttm_income_statement))).sum()
    interest_expense = ttm_income_statement.get("Interest Expense", pd.Series([0] * len(ttm_income_statement))).sum()
    book_value_of_equity = last_balance_sheet.get("Stockholders Equity", pd.Series([0])).iloc[0]
    book_value_of_debt = last_balance_sheet.get("Total Debt", pd.Series([0])).iloc[0]
    cash_and_marketable_securities = last_balance_sheet.get("Cash Cash Equivalents And Short Term Investments", pd.Series([0])).iloc[0]
    cross_holdings_and_other_non_operating_assets = last_balance_sheet.get("Minority Interest", pd.Series([0])).iloc[0]
    number_of_shares_outstanding = info.get("sharesOutstanding", 0)
    curr_price = info.get("previousClose", 0)
    effective_tax_rate = (ttm_income_statement.get("Tax Rate For Calcs", pd.Series([0])) * ttm_income_statement.get("Pretax Income", pd.Series([0]))).sum() / ttm_income_statement.get("Pretax Income", pd.Series([1])).sum()

    regions = region_mapper.get_closest(info["country"])

    industry = info["industry"]
    avg_betas = avg_metrics["Unlevered Beta"]
    unlevered_beta, industry = get_industry_beta(industry, industry_mapper, avg_betas)

    marketscreener_url = get_marketscreener_url(info["symbol"])

    regional_revenues = get_revenue_by_region(info["symbol"], marketscreener_url)
    equity_risk_premium, mapped_regional_revenues = get_regional_crps(regional_revenues, region_mapper, country_erps)
    equity_risk_premium = equity_risk_premium + mature_erp
    _, company_spread, prob_of_failure = synthetic_rating(info["marketCap"], ttm_income_statement["Operating Income"].sum(), interest_expense)
    pre_tax_cost_of_debt = risk_free_rate + company_spread + country_erps[regions[0]]

    revenue_growth_rate_next_year, compounded_annual_revenue_growth_rate = get_revenue_forecasts(info["symbol"], marketscreener_url)
    operating_margin_next_year = ttm_income_statement["Operating Income"].sum() / revenues
    target_pre_tax_operating_margin = avg_metrics["Pre-tax Operating Margin (Unadjusted)"][industry]
    target_pre_tax_operating_margin = max(target_pre_tax_operating_margin, operating_margin_next_year)
    year_of_convergence_for_margin = 5
    years_of_high_growth = 5
    curr_sales_to_capital_ratio = revenues / (book_value_of_equity + book_value_of_debt - cash_and_marketable_securities - cross_holdings_and_other_non_operating_assets)
    sales_to_capital_ratio_early = curr_sales_to_capital_ratio
    sales_to_capital_ratio_steady = avg_metrics["Sales/Capital"][industry]
    return {
        "name": name,
        "revenues": revenues,
        "interest_expense": interest_expense,
        "book_value_of_equity": book_value_of_equity,
        "book_value_of_debt": book_value_of_debt,
        "cash_and_marketable_securities": cash_and_marketable_securities,
        "cross_holdings_and_other_non_operating_assets": cross_holdings_and_other_non_operating_assets,
        "number_of_shares_outstanding": number_of_shares_outstanding,
        "curr_price": curr_price,
        "effective_tax_rate": effective_tax_rate,
        "marginal_tax_rate": marginal_tax_rate,
        "unlevered_beta": unlevered_beta,
        "risk_free_rate": risk_free_rate,
        "equity_risk_premium": equity_risk_premium,
        "mature_erp": mature_erp,
        "pre_tax_cost_of_debt": pre_tax_cost_of_debt,
        "average_maturity": average_maturity,
        "prob_of_failure": prob_of_failure,
        "value_of_options": value_of_options,
        "revenue_growth_rate_next_year": revenue_growth_rate_next_year,
        "operating_margin_next_year": operating_margin_next_year,
        "compounded_annual_revenue_growth_rate": compounded_annual_revenue_growth_rate,
        "target_pre_tax_operating_margin": target_pre_tax_operating_margin,
        "year_of_convergence_for_margin": year_of_convergence_for_margin,
        "years_of_high_growth": years_of_high_growth,
        "sales_to_capital_ratio_early": sales_to_capital_ratio_early,
        "sales_to_capital_ratio_steady": sales_to_capital_ratio_steady,
        "extras": {
            "regional_revenues": regional_revenues,
            "industry": industry,
            "historical_revenue_growth": info.get("revenueGrowth", 0),
            "mapped_regional_revenues": mapped_regional_revenues,
        },
    }


def main():
    # url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
    # response = requests.get(url)
    # file_content = response.text
    # tickers = file_content.split("\n")
    # print("Number of tickers:", len(tickers))
    # tickers = ["AAPL"]
    tickers = []
    country_erps = get_country_erp()
    region_mapper = StringMapper(list(country_erps.keys()))
    avg_metrics = get_industry_avgs()
    avg_betas = avg_metrics["Unlevered Beta"]
    industry_mapper = StringMapper(list(avg_betas.keys()), threshold=0.7)
    risk_free_rate = get_10year_tbill()
    mature_erp = get_mature_erp()

    uri = f"mongodb+srv://{os.getenv('MONGODB_USERNAME')}:{os.getenv('MONGODB_DB_PASSWORD')}@{os.getenv('MONGODB_DB_NAME')}.kdnx4hj.mongodb.net/?retryWrites=true&w=majority&appName={os.getenv('MONGODB_DB_NAME')}"
    client = MongoClient(uri, server_api=ServerApi("1"))
    db = client[os.getenv("MONGODB_DB_NAME")]["dcf_inputs"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, ticker in enumerate(tickers):
            if i > 0 and i % MAX_WORKERS == 0:
                time.sleep(1)
            future = executor.submit(process_ticker, ticker, country_erps, region_mapper, avg_metrics, industry_mapper, mature_erp, risk_free_rate, db)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"[ERROR] - {e}")

    # Write all constants to the database
    db.update_one(
        filter={},
        update={
            "$set": {
                "Macro": {
                    "Country ERPs": country_erps,
                    "Avg Metrics": avg_metrics,
                    "Risk Free Rate": risk_free_rate,
                    "Mature ERP": mature_erp,
                }
            }
        },
        upsert=True,
    )


def process_ticker(ticker, country_erps, region_mapper, avg_metrics, industry_mapper, mature_erp, risk_free_rate, db):
    url = get_marketscreener_url(ticker)
    if url is None:
        return f"[INFO] Could not find URL for {ticker}"
    try:
        dcf_inputs = get_dcf_inputs(ticker, country_erps, region_mapper, avg_metrics, industry_mapper, mature_erp, risk_free_rate)
        dcf_inputs = json.dumps(dcf_inputs, cls=CustomEncoder)
        dcf_inputs = json.loads(dcf_inputs)
        db.update_one({"Ticker": ticker}, {"$set": dcf_inputs}, upsert=True)
        return f"[SUCCESS] {ticker} - DCF inputs updated"
    except Exception as e:
        return f"[ERROR] {ticker} - {e}"


if __name__ == "__main__":
    main()
