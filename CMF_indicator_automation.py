import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import numpy as np
from vnstock import Vnstock

# NEW: import tqdm
from tqdm import tqdm

###############################################################################
# 1. Logging / Display Configuration
###############################################################################

# logging.getLogger("vnstock").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

pd.set_option('display.max_rows', None)  # Display all rows


###############################################################################
# 2. Function Definitions
###############################################################################

def fetch_price_history(ticker, start_date=None, end_date=None):
    """
    Fetch historical price data for a given ticker using Vnstock, 
    from start_date to end_date. Returns a DataFrame with time, high, low, 
    close, volume columns.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    try:
        # Fetch data
        stock_obj = Vnstock().stock(symbol=ticker, source='VCI')
        df = stock_obj.quote.history(start=start_date, end=end_date, to_df=True)

        if df.empty:
            return pd.DataFrame()

        # Convert time to datetime, sort descending
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time', ascending=False)

        # # Pause to avoid rate-limit issues
        # time.sleep(1)

        return df[['time', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()


def fetch_data_for_ticker(ticker, start_date, end_date):
    """
    Fetch data for a single ticker, returning a DataFrame with [time, high, low, 
    close, volume, symbol].
    """
    try:
        df = fetch_price_history(ticker, start_date, end_date)
        if not df.empty:
            df['symbol'] = ticker
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def calculate_cmf(df, period=20):
    """
    Calculate Chaikin Money Flow (CMF) for a single symbol DataFrame.
    - Sorts by time ascending
    - Applies the standard CMF formula
    - Returns DataFrame sorted descending by time with a 'cmf' column.
    """
    # Sort oldest->newest
    df = df.sort_values(by='time', ascending=True)

    # Money Flow Multiplier (MFM)
    df['mfm'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (
        df['high'] - df['low']
    ).replace(0, np.nan)

    # Money Flow Volume (MFV)
    df['mfv'] = df['mfm'] * df['volume']

    # Rolling sums for CMF
    df['cmf'] = (
        df['mfv'].rolling(window=period).sum() /
        df['volume'].rolling(window=period).sum()
    )

    # Sort newest->oldest before returning
    df = df.sort_values(by='time', ascending=False)
    return df


def check_cmf_condition(df):
    """
    Add columns:
        - min_last_5_cmf: rolling min of the last 5 CMF values
        - cmf_diff: difference between cmf and min_last_5_cmf
        - is_cmf_condition_met: True if cmf_diff > 0.2 and min_last_5_cmf not NaN
    Returns the modified DataFrame sorted ascending.
    """
    # Sort ascending to have a proper rolling window over the correct chronological order
    df = df.sort_values(by='time', ascending=True)

    df['min_last_5_cmf'] = df['cmf'].rolling(window=5).min()
    df['cmf_diff'] = df['cmf'] - df['min_last_5_cmf']
    df['is_cmf_condition_met'] = (df['cmf_diff'] > 0.2) & df['min_last_5_cmf'].notna()

    return df


###############################################################################
# 3. Main Workflow
###############################################################################

def main():
    # Print message to indicate start of the program
    print("Starting CMF Indicator Automation...")

    # Define your date range
    start_date = (datetime.today() - pd.DateOffset(days=40)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Get symbol list from Vnstock.
    print("Fetching symbol list...")
    stock = Vnstock().stock(source='VCI')
    df_symbols = stock.listing.symbols_by_exchange()
    filtered_df = df_symbols[df_symbols['exchange'].isin(['HSX', 'HNX'])]
    symbol_list = filtered_df['symbol'].tolist()
    print("Number of symbols:", len(symbol_list))
    print("Symbols List:", symbol_list)
    print("=====================================")

    # For demonstration, just do it manually:
    # symbol_list = ['VNM', 'PTB']  # or any other set of symbols

    # # 3.1 Parallel data fetching
    # price_data_list = []
    # num_cores = os.cpu_count()

    # with ThreadPoolExecutor(max_workers=num_cores) as executor:
    #     # Schedule the fetches
    #     future_to_ticker = {
    #         executor.submit(fetch_data_for_ticker, ticker, start_date, end_date): ticker
    #         for ticker in symbol_list
    #     }
    #     for future in as_completed(future_to_ticker):
    #         ticker = future_to_ticker[future]
    #         try:
    #             data = future.result()
    #             if not data.empty:
    #                 price_data_list.append(data)
    #         except Exception as e:
    #             print(f"Failed to process {ticker}: {e}")

    # # Combine all DataFrames
    # if not price_data_list:
    #     print("No data fetched.")
    #     return

    # price_data = pd.concat(price_data_list, ignore_index=True)

    ###########################################################################
    # 3.1 Parallel data fetching with progress bar
    ###########################################################################
    price_data_list = []
    num_cores = os.cpu_count()

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        future_to_ticker = {
            executor.submit(fetch_data_for_ticker, ticker, start_date, end_date): ticker
            for ticker in symbol_list
        }
        
        # 'as_completed' yields futures as they finish
        # We wrap it in 'tqdm' to track how many have completed
        for future in tqdm(as_completed(future_to_ticker), 
                           total=len(symbol_list), 
                           desc="Fetching data"):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if not data.empty:
                    price_data_list.append(data)
            except Exception as e:
                print(f"Failed to process {ticker}: {e}")

    # Combine all DataFrames
    if not price_data_list:
        print("No data fetched.")
        return

    price_data = pd.concat(price_data_list, ignore_index=True)    
    print("Data fetched successfully.")
    print("=====================================")

    # 3.2 Calculate CMF
    # Explicitly pick the columns needed, then apply your CMF function
    price_data = (
        price_data
        .groupby('symbol', group_keys=False)[['time', 'symbol', 'high', 'low', 'close', 'volume']]
        .apply(calculate_cmf)
    )

    # 3.3 Drop unnecessary columns
    price_data.drop(columns=['high', 'low', 'close', 'volume', 'mfm', 'mfv'], inplace=True)

    # 3.4 Check CMF conditions
    price_data = (
        price_data
        .groupby('symbol', group_keys=False)[['symbol', 'time', 'cmf']]
        .apply(check_cmf_condition)
    )

    # 3.5 Get latest row per symbol
    latest_data = (
        price_data
        .sort_values(by='time', ascending=False)
        .drop_duplicates(subset=['symbol'])
    )

    # Show the final columns
    print(latest_data[['symbol', 'time', 'cmf', 'min_last_5_cmf', 'is_cmf_condition_met']])
    print("=====================================")

    # Identify symbols that meet the condition
    symbols_to_buy = latest_data[latest_data['is_cmf_condition_met']]['symbol'].tolist()
    print('Symbols to buy:', symbols_to_buy)


###############################################################################
# 4. Entry Point
###############################################################################

if __name__ == "__main__":
    main()
