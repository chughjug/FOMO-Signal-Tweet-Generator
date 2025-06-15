import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import time
import numpy as np
import os
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Selenium WebDriver setup with headless mode
def setup_driver():
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Use Chromium in GitHub Actions environment
        if os.getenv('GITHUB_ACTIONS') == 'true':
            chromium_path = shutil.which('chromium-browser') or shutil.which('chromium')
            if chromium_path:
                chrome_options.binary_location = chromium_path
                logging.info(f"Using Chromium binary at: {chromium_path}")
            else:
                logging.warning("Chromium not found, using default Chrome path")
        
        driver = webdriver.Chrome(options=chrome_options)
        logging.info("Headless Selenium WebDriver initialized successfully")
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize Selenium WebDriver: {e}")
        raise

# Scrape Yahoo Finance most active tickers with Selenium (original working version)
def get_yahoo_high_volume_tickers(driver):
    try:
        url = "https://finance.yahoo.com/screener/predefined/most_actives?count=20"
        driver.get(url)
        logging.info(f"Navigating to {url}")
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.yf-1570k0a'))
        )
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', class_='yf-1570k0a')
        if not table:
            logging.error("Table with class 'yf-1570k0a' not found")
            return {}
        tickers = {}
        rows = table.find('tbody').find_all('tr', class_='row yf-1570k0a')
        for row in rows:
            ticker_span = row.find('span', class_='symbol yf-1jsynna')
            if ticker_span:
                ticker = ticker_span.text.strip()
                if re.match(r'^[A-Z]{1,5}$', ticker):
                    tickers[ticker] = "Most Active"
        logging.info(f"Extracted {len(tickers)} Yahoo Finance most active tickers: {list(tickers.keys())}")
        return tickers
    except Exception as e:
        logging.error(f"Yahoo Finance most active error: {e}")
        return {}

# Scrape Yahoo Finance trending tickers with Selenium (EXACT DUPLICATE of the most active function)
def get_yahoo_trending_tickers(driver):
    try:
        url = "https://finance.yahoo.com/markets/stocks/trending/"
        driver.get(url)
        logging.info(f"Navigating to {url}")
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.yf-1570k0a'))
        )
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', class_='yf-1570k0a')
        if not table:
            logging.error("Table with class 'yf-1570k0a' not found")
            return {}
        tickers = {}
        rows = table.find('tbody').find_all('tr', class_='row yf-1570k0a')
        for row in rows:
            ticker_span = row.find('span', class_='symbol yf-1jsynna')
            if ticker_span:
                ticker = ticker_span.text.strip()
                if re.match(r'^[A-Z]{1,5}$', ticker):
                    tickers[ticker] = "Trending"
        logging.info(f"Extracted {len(tickers)} Yahoo Finance trending tickers: {list(tickers.keys())}")
        return tickers
    except Exception as e:
        logging.error(f"Yahoo Finance trending error: {e}")
        return {}

# Analyze ticker with comprehensive metrics
def analyze_ticker(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        if data.empty or len(data) < 10:
            logging.warning(f"Insufficient data for {ticker}: {data.shape}, dates {data.index[0] if not data.empty else None} to {data.index[-1] if not data.empty else None}")
            return None
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(col in data.columns for col in expected_columns):
            logging.error(f"Unexpected columns for {ticker}: {data.columns}")
            return None
        logging.info(f"Data for {ticker}: shape {data.shape}, columns {data.columns}, dates {data.index[0]} to {data.index[-1]}")
        price_change_2d = ((data['Close'].iloc[-1] - data['Close'].iloc[-3]) / data['Close'].iloc[-3]) * 100 if len(data) >= 3 else np.nan
        price_change_1d = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100 if len(data) >= 2 else np.nan
        vol_2d_avg = data['Volume'].tail(2).mean() if len(data) >= 2 else np.nan
        vol_10d_avg = data['Volume'].tail(10).mean() if len(data) >= 10 else np.nan
        vol_20d_avg = data['Volume'].iloc[-21:-1].mean() if len(data) >= 21 else np.nan
        vol_ratio_2d_10d = vol_2d_avg / vol_10d_avg if vol_10d_avg > 0 else np.nan
        vol_ratio_2d_20d = vol_2d_avg / vol_20d_avg if vol_20d_avg > 0 else np.nan
        vol_spike = data['Volume'].iloc[-1] > 2 * vol_20d_avg if vol_20d_avg > 0 else False
        rsi = RSIIndicator(close=data['Close'], window=14).rsi().iloc[-1] if len(data) >= 15 and data['Close'].notna().sum() >= 15 else np.nan
        up_days = (data['Close'].diff() > 0).rolling(window=len(data)).sum().iloc[-1] if len(data) > 1 else 0
        tr = np.maximum(data['High'] - data['Low'], 
                        np.maximum(abs(data['High'] - data['Close'].shift(1)), 
                                   abs(data['Low'] - data['Close'].shift(1))))
        atr = tr.rolling(window=14).mean().iloc[-1] if len(tr) >= 14 else np.nan
        return {
            'Ticker': ticker,
            '2D_Price_Change_%': round(price_change_2d, 2) if not np.isnan(price_change_2d) else None,
            '1D_Price_Change_%': round(price_change_1d, 2) if not np.isnan(price_change_1d) else None,
            '2D_vs_10D_Vol_Ratio': round(vol_ratio_2d_10d, 2) if not np.isnan(vol_ratio_2d_10d) else None,
            '2D_vs_20D_Vol_Ratio': round(vol_ratio_2d_20d, 2) if not np.isnan(vol_ratio_2d_20d) else None,
            'Vol_Spike': vol_spike,
            'RSI': round(rsi, 2) if not np.isnan(rsi) else None,
            'Consecutive_Up_Days': int(up_days) if not np.isnan(up_days) else 0,
            'ATR': round(atr, 2) if not np.isnan(atr) else None
        }
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")
        return None

# FOMO detection with specific triggers
def detect_fomo(stock_data):
    triggers = []
    
    if stock_data['2D_Price_Change_%'] is not None and stock_data['2D_Price_Change_%'] > 10:
        triggers.append(f"🚀 2D Price Surge (+{stock_data['2D_Price_Change_%']:.1f}%)")
    
    if stock_data['RSI'] is not None and stock_data['RSI'] > 70:
        triggers.append(f"📈 Overbought (RSI {stock_data['RSI']:.0f})")
    
    if stock_data['2D_vs_10D_Vol_Ratio'] is not None and stock_data['2D_vs_10D_Vol_Ratio'] > 1.5:
        triggers.append(f"📊 Volume Spike ({stock_data['2D_vs_10D_Vol_Ratio']:.1f}x normal)")
    
    if stock_data['1D_Price_Change_%'] is not None and stock_data['1D_Price_Change_%'] > 5:
        triggers.append(f"💨 Strong Momentum (+{stock_data['1D_Price_Change_%']:.1f}% today)")
    
    if stock_data['Vol_Spike']:
        triggers.append("🔊 Heavy Trading Volume")
    
    return triggers

# Main execution
def main():
    print("🚀 Starting ticker scraping...")
    
    # Create data directory at the beginning to ensure it exists
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    logging.info(f"Created data directory at: {os.path.abspath(data_dir)}")
    
    # Create a single driver instance
    driver = setup_driver()
    
    try:
        # Scrape both sites using the same driver
        print("\n🔍 Scraping most active tickers...")
        most_active_tickers = get_yahoo_high_volume_tickers(driver)
        
        print("\n🔍 Scraping trending tickers...")
        trending_tickers = get_yahoo_trending_tickers(driver)
        
        # Combine tickers from both sources
        all_tickers = {**most_active_tickers, **trending_tickers}
        ticker_list = list(all_tickers.keys())
        
        # Only use fallback if we got no tickers from either source
        if not ticker_list:
            logging.warning("No tickers found, using fallback tickers.")
            ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        
        print("\n📊 Scraping Results:")
        print(f"Most Active Tickers: {', '.join(most_active_tickers.keys()) if most_active_tickers else 'None'}")
        print(f"Trending Tickers: {', '.join(trending_tickers.keys()) if trending_tickers else 'None'}")
        print(f"✅ Total Unique Tickers: {len(ticker_list)}")
        
        print("\n🔍 Starting technical analysis...")
        results = []
        for i, ticker in enumerate(ticker_list, 1):
            print(f"Analyzing {i}/{len(ticker_list)}: {ticker}")
            if result := analyze_ticker(ticker):
                results.append(result)
        
        # Create filename with current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_path = os.path.join(data_dir, f"fomo_alerts_{current_date}.txt")
        
        if results:
            alert_data = []
            
            for r in results:
                # Check for FOMO signals
                if triggers := detect_fomo(r):
                    alert_data.append({
                        'ticker': r['Ticker'],
                        'triggers': triggers
                    })
            
            # Save to file with trigger explanations and alerts
            with open(output_path, 'w') as f:
                # Write trigger explanations
                f.write("FOMO TRIGGER EXPLANATIONS:\n")
                f.write("🚀 2D Price Surge (>10% gain)\n")
                f.write("📈 Overbought (RSI >70)\n")
                f.write("📊 Volume Spike (2-day vol >1.5x 10-day avg)\n")
                f.write("💨 Strong Momentum (>5% daily gain)\n")
                f.write("🔊 Heavy Trading Volume (today >2x 20-day avg)\n\n")
                
                if alert_data:
                    f.write("FOMO ALERTS:\n")
                    for alert in alert_data:
                        alert_line = f"🔥 {alert['ticker']} ALERT: {' | '.join(alert['triggers'])}"
                        f.write(alert_line + "\n")
                    f.write(f"\nTotal FOMO alerts: {len(alert_data)}")
                else:
                    f.write("No FOMO alerts detected today")
            
            # Print results to console
            if alert_data:
                print("\n🔥 FOMO ALERTS DETECTED:")
                for alert in alert_data:
                    print(f"🔥 {alert['ticker']} ALERT: {' | '.join(alert['triggers'])}")
                print(f"\n💾 Saved {len(alert_data)} FOMO alerts to {output_path}")
            else:
                print("\n😢 No stocks with FOMO signals found")
        else:
            # Create empty file if no results
            with open(output_path, 'w') as f:
                f.write("No valid analysis results for today\n")
            print("\n❌ No valid analysis results")
    
    except Exception as e:
        logging.error(f"Main execution error: {e}")
    
    finally:
        # Ensure the driver is closed even if errors occur
        driver.quit()
        logging.info("WebDriver closed")

if __name__ == "__main__":
    main()
