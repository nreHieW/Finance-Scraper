import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
import json
import os
import gspread
import requests
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from concurrent.futures import ThreadPoolExecutor


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


MAX_WORKERS = 100
header = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36"}


def fetch_html(url):
    response = requests.get(url, headers=header)
    return response.text


def get_htmls(urls):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        html_responses = list(executor.map(fetch_html, urls))
    return html_responses


def get_marketscreener_links(tickers):
    search_queries = ["https://www.marketscreener.com/search/?q=" + "+".join(x.split()) for x in tickers]
    found_htmls = get_htmls(search_queries)
    links = {}
    for ticker, html in zip(tickers, found_htmls):
        soup = BeautifulSoup(html, features="lxml")
        rows = soup.findAll("tr")
        for row in rows:
            currency_tag = row.find("span", {"class": "txt-muted"})
            if currency_tag:
                currency = currency_tag.text.strip()
                if currency == "USD":
                    link = row.find("a", href=True)["href"]
                    links[ticker] = "https://www.marketscreener.com" + link + "finances/"
                    break
    return links


def parse_marketscreener(marketscreener_urls):
    htmls = get_htmls(list(marketscreener_urls.values()))
    htmls = dict(zip(marketscreener_urls.keys(), htmls))
    dfs = []
    for ticker, html in htmls.items():
        try:
            soup = BeautifulSoup(html, features="lxml")
            for div in soup.find_all("div", {"class": "card card--collapsible mb-15"}):
                header_text = div.find("div", {"class": "card-header"}).text.lower()
                if "income statement" in header_text and "annual" in header_text:
                    income_statement = pd.read_html(str(div.find("table")))[0]
                    income_statement = income_statement.dropna(axis=1, how="all")
                    income_statement.iloc[:, 0] = income_statement.iloc[:, 0].str.replace(r"\d", "", regex=True)
                    income_statement.set_index(income_statement.columns[0], inplace=True)
                    income_statement.index = income_statement.index.str.strip()
                    indiv = income_statement.loc[["Net sales", "Net income", "EBITDA", "EBIT"]]
                    indiv = indiv.stack()
                    indiv.index = [" ".join(x) for x in indiv.index]
                    indiv = indiv.to_frame().T
                    indiv["Currency"] = div.find_all("sup")[0].attrs["title"].strip().split()[0]
                    indiv["Unit"] = div.find_all("sup")[0].attrs["title"].strip().split()[-1]
                    indiv["Ticker"] = ticker
                    dfs.append(indiv)
                    break
        except Exception as e:
            print("Error parsing", ticker, e)
    marketscreener = pd.concat(dfs, axis=0, join="outer", ignore_index=True)
    return marketscreener.reset_index(drop=True).fillna(0).set_index("Ticker")


def get_info(ticker):
    try:
        ticker_info = yf.Ticker(ticker).info
        return {
            "Ticker": ticker,
            "Name": ticker_info.get("longName"),
            "Market Cap": ticker_info.get("marketCap"),
            "Sector": ticker_info.get("sector"),
            "Summary": ticker_info.get("longBusinessSummary"),
            "Industry": ticker_info.get("industry"),
            "Shares Outstanding": ticker_info.get("sharesOutstanding"),
            "Institution Ownership": ticker_info.get("heldPercentInstitutions"),
            "Price": ticker_info.get("currentPrice"),
            "52-Week High": ticker_info.get("fiftyTwoWeekHigh"),
            "52-Week Low": ticker_info.get("fiftyTwoWeekLow"),
            "Enterprise Value": ticker_info.get("enterpriseValue"),
            "Beta": ticker_info.get("beta"),
        }
    except Exception as e:
        # print("Error fetching", ticker, e)
        return {
            "Ticker": ticker,
            "Name": 0,
            "Market Cap": 0,
            "Sector": 0,
            "Summary": 0,
            "Industry": 0,
            "Shares Outstanding": 0,
            "Institution Ownership": 0,
            "Price": 0,
            "52-Week High": 0,
            "52-Week Low": 0,
            "Enterprise Value": 0,
            "Beta": 0,
        }


def get_and_parse_yahoo(tickers):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(get_info, tickers))

    return pd.DataFrame(results).set_index("Ticker")


def parse_finviz(tickers):
    finviz_urls = ["https://finviz.com/quote.ashx?t=" + x for x in tickers]
    htmls = get_htmls(finviz_urls)
    perf_columns = [
        "Perf Week",
        "Perf Month",
        "Perf Quarter",
        "Perf Half Y",
        "Perf Year",
        "Perf YTD",
    ]
    dfs = []
    for i, html in enumerate(htmls):
        try:
            soup = BeautifulSoup(html, "lxml")
            table = soup.find("table", class_="snapshot-table2")
            if table:
                df = pd.read_html(str(table))[0].iloc[:, -2:].set_index(10)
                df[11] = df[11].str.replace("%", "")
                perf_values = df.loc[perf_columns].astype(float).T.reset_index(drop=True)
                indiv = pd.DataFrame(perf_values, columns=perf_columns)
            else:
                indiv = pd.DataFrame([[0] * len(perf_columns)], columns=perf_columns)
            indiv["Ticker"] = tickers[i]
            dfs.append(indiv)
        except Exception as e:
            print("Error parsing", tickers[i], e)
    return pd.concat(dfs, axis=0, ignore_index=True).set_index("Ticker")


if __name__ == "__main__":

    cred = json.loads(os.environ.get("GOOGLE_CREDS"))
    gc = gspread.service_account_from_dict(cred)
    wb = gc.open(os.environ.get("GOOGLE_SHEET_NAME"))

    data_sheet = wb.worksheet("Data")

    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
    response = requests.get(url)
    file_content = response.text
    tickers = file_content.split("\n")
    print("Number of tickers:", len(tickers))

    found_links = get_marketscreener_links(tickers)
    marketscreener_df = parse_marketscreener(found_links)
    yahoo_df = get_and_parse_yahoo(tickers)
    finviz_df = parse_finviz(tickers)

    df = pd.concat([yahoo_df, marketscreener_df, finviz_df], axis=1, ignore_index=True)
    df.columns = yahoo_df.columns.tolist() + marketscreener_df.columns.tolist() + finviz_df.columns.tolist()
    df.reset_index(inplace=True)
    df.sort_values("Ticker", inplace=True)
    df.fillna(0, inplace=True)
    data_sheet.clear()
    data_sheet.update([df.columns.values.tolist()] + df.values.tolist())

    uri = f"mongodb+srv://{os.getenv('MONGODB_USERNAME')}:{os.getenv('MONGODB_DB_PASSWORD')}@{os.getenv('MONGODB_DB_NAME')}.g29k6mj.mongodb.net/?retryWrites=true&w=majority&appName={os.getenv('MONGODB_DB_NAME')}"
    client = MongoClient(uri, server_api=ServerApi("1"))
    db = client[os.getenv("MONGODB_DB_NAME")]["financials"]
    data = df.to_dict(orient="records")
    data = json.loads(json.dumps(data, cls=CustomEncoder))
    for record in data:
        db.update_one({"Ticker": record["Ticker"]}, {"$set": record}, upsert=True)
