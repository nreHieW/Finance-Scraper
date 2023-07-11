

import yfinance as yf
import pandas as pd 
from bs4 import BeautifulSoup
import json
import aiohttp
import asyncio
import os
import gspread
import numpy as np
from concurrent.futures import ThreadPoolExecutor
pd.options.mode.chained_assignment = None  # default='warn'

# async works only on finviz and marketscreener. yfinance is written synchonously

years = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']
col_names = ['Ticker',
    'Name', 'Market Cap', 'Sector', 'Business Summary', 'Industry', 'Shares Outstanding',
    'Institution Ownership', 'Current Price', '52-Week High', '52-Week Low', 'Enterprise Value', 'Beta',
    'Weekly Price Change %', 'Monthly Price Change %',
    'Quarterly Price Change %', 'Half-Yearly Price Change %', 'Yearly Price Change %',
    'YTD Price Change %'
]

col_names += [f'{year} Revenues' for year in years]
col_names += [f'{year} Net Income' for year in years]
col_names += [f'{year} EBITDA' for year in years]
col_names += [f'{year} EBIT' for year in years]
col_names += ['Currency', 'Unit']

MAX_WORKERS = 10

async def get_page(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as r:
        return await r.text()

async def visit_all_links(session: aiohttp.ClientSession, urls: list, sem: asyncio.Semaphore) -> list:
    task_lst = []
    for url in urls:
        async with sem:
            task = asyncio.create_task(get_page(session, url))
            task_lst.append(task)
    results = await asyncio.gather(*task_lst)
    return results

async def get_all_html(urls: list, max_workers: int = MAX_WORKERS) -> list:
    header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'}
    sem = asyncio.Semaphore(max_workers)
    async with aiohttp.ClientSession(headers=header) as session:
        data = await visit_all_links(session, urls, sem)
        return data

def parse_marketscreener(results):

    ### HELPER FUNCTIONS
    def fix_names(x):
        lst = ['1','2','3']
        for num in lst:
            if num in x:
                val = x.replace(num,"")
                return val
            else:
                val = x
        return val
    def process_df(df):
        df = df.dropna(axis=1,how='all')
        df.iloc[:,0] = df.iloc[:,0].apply(fix_names)
        df.set_index(df.columns[0],inplace=True)
        df.index = df.index.str.strip()
        return df

    marketscreener = pd.DataFrame()
    for html in results:
        divs = BeautifulSoup(html,features="lxml").find_all("div", {"class": "card card--collapsible mb-15"})
        for div in divs:
            header_text = div.find("div", {"class": "card-header"}).text
            if ("income statement" in header_text.lower()) and ("annual" in header_text.lower()):
                income_statement = div.find("table")
                income_statement = pd.read_html(str(income_statement))[0]
                currency = div.find_all("sup")[0].attrs["title"].replace("\n", "").strip().split(" ")[0]
                unit = div.find_all("sup")[0].attrs["title"].replace("\n", "").strip().split(" ")[-1]
                break
        currency, unit = pd.Series(currency), pd.Series(unit)
        income_statement = process_df(income_statement)
        col_names_is = income_statement.columns.tolist()
        applicable = [col for col in col_names_is if col in years]
        remain = [col for col in years if col not in applicable]
        metrics = []
        filler = pd.Series([0] * len(remain),dtype='object')
        metrics_names = ["Net sales", "Net income", "EBITDA", "EBIT"]
        for metric in metrics_names:
            metrics.append(pd.Series(income_statement.loc[:,applicable].loc[metric]).T.reset_index(drop=True).str.replace(' ',''))
            metrics.append(filler)
        indiv = pd.DataFrame(pd.concat(metrics)).T
        indiv = pd.concat([indiv,currency,unit],axis=1)
        indiv.columns = list(range(len(indiv.columns.tolist()))) # Net sales, Net income, EBITDA, EBIT for X years, currency, unit
        marketscreener = pd.concat([marketscreener,indiv],axis=0)
    return marketscreener.reset_index(drop=True).replace("-", 0)

def get_and_parse_yahoo(tickers):
    def get_ticker(x):
        ticker = yf.Ticker(x)
        return ticker.info
    yahoo = pd.DataFrame()
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(get_ticker, tickers))

    for idx, info in enumerate(results):
        data = [
            {'Ticker': tickers[idx]},
            {'Name': info.get('longName')},
            {'Market Cap': info.get('marketCap')},
            {'Sector': info.get('sector')},
            {'Summary': info.get('longBusinessSummary')},
            {'Industry': info.get('industry')},
            {'Shares Outstanding': info.get('sharesOutstanding')},
            {'Institution Ownership': info.get('heldPercentInstitutions')},
            {'Price': info.get('currentPrice')},
            {'52-Week High': info.get('fiftyTwoWeekHigh')},
            {'52-Week Low': info.get('fiftyTwoWeekLow')},
            {'Enterprise Value': info.get('enterpriseValue')},
            {'Beta': info.get('beta')},
        ]
        indiv = pd.DataFrame([x.values() for x in data],index=[list(x.keys())[0] for x in data]).T.reset_index(drop=True)
        yahoo = pd.concat([yahoo, indiv], axis=0)

    return yahoo.reset_index(drop=True)

def parse_finviz(results):
    finviz = pd.DataFrame()
    perf_columns = ['Perf Week', 'Perf Month', 'Perf Quarter', 'Perf Half Y', 'Perf Year', 'Perf YTD']

    for html in results:
        soup = BeautifulSoup(html, features="lxml")
        table = soup.find_all('table', {'class': 'snapshot-table2'})
        try:
            df = pd.read_html(str(table))[0].iloc[:,-2:].set_index(10)
            df[11] = df[11].str.replace('%', '')
            perf_values = df.loc[perf_columns].astype(float).T.reset_index(drop=True)
            indiv = pd.DataFrame(perf_values, columns=perf_columns)
        except:
            indiv = pd.DataFrame([[0] * 6], columns=perf_columns)
        finviz = pd.concat([finviz, indiv], axis=0, ignore_index=True)
    return finviz

async def get_marketscreener_links(tickers, names, links):
    tickers = [x for x in tickers if x not in links]
    html_list = await get_all_html(['https://www.marketscreener.com/search/?q=' + "+".join(x.split()) for x in names if x not in links])
    for ticker, html in zip(tickers,html_list):
        soup = BeautifulSoup(html,features="lxml")
        rows = soup.findAll("tr")
        for row in rows:
            currency_tag = row.find('span', {"class": "txt-muted"})
            if currency_tag:
                currency = currency_tag.text.strip()
                if currency == 'USD':
                    link = row.find('a', href=True)['href']
                    links[ticker] = "https://www.marketscreener.com" + link + 'finances/'
                    break
        
        with open('Lookup/MarketScreener.json','w') as f:
            json.dump(links,f)
    return links

if __name__ == "__main__":
    cred = json.loads(os.environ.get('GOOGLE_CREDS'))
    gc = gspread.service_account_from_dict(cred)
    wb = gc.open(os.environ.get('GOOGLE_SHEET_NAME'))
    data_sheet = wb.worksheet("Data")
    tickers = data_sheet.col_values(1)[1:]
    tickers = [x.upper() for x in tickers]

    yahoo = get_and_parse_yahoo(tickers)
    names = yahoo['Name'].tolist()

    if os.path.exists('Lookup/Marketscreener.json'):
        with open('Lookup/Marketscreener.json','r') as f:
            links = json.load(f)
    else:
        links = dict()

    links = asyncio.run(get_marketscreener_links(tickers, names,links))
    market_screener_urls = [links[x] for x in tickers if x in links]
    market_screener_htmls = asyncio.run(get_all_html(market_screener_urls))
    marketscreener = parse_marketscreener(market_screener_htmls)
    finviz_urls = ['https://finviz.com/quote.ashx?t='+ x for x in tickers]
    finviz_htmls = asyncio.run(get_all_html(finviz_urls))
    finviz = parse_finviz(finviz_htmls)

    final = pd.concat([yahoo,finviz,marketscreener],axis=1)
    final = final.apply(pd.to_numeric, errors = "ignore")
    final = final.replace(np.nan, 0).fillna(0)
    final.columns = col_names
    data_sheet.update([final.columns.values.tolist()] + final.values.tolist())



