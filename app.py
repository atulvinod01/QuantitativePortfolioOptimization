import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="Quant Portfolio Optimizer", layout="wide")
DB_FILE = "portfolio_data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker TEXT,
            date DATE,
            adj_close REAL,
            PRIMARY KEY (ticker, date)
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(df, ticker):
    conn = sqlite3.connect(DB_FILE)
    data = df.reset_index()[['Date', 'Adj Close']].copy()
    data.columns = ['date', 'adj_close']
    data['ticker'] = ticker
    data['date'] = data['date'].dt.date
    
    c = conn.cursor()
    c.execute("DELETE FROM stock_prices WHERE ticker = ?", (ticker,))
    
    data.to_sql('stock_prices', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()

def load_from_db(ticker):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT date, adj_close FROM stock_prices WHERE ticker = ? ORDER BY date"
    df = pd.read_sql(query, conn, params=(ticker,), parse_dates=['date'])
    conn.close()
    if not df.empty:
        df.set_index('date', inplace=True)
    return df

def get_data(tickers, start_date, end_date):
    init_db()
    combined_data = pd.DataFrame()
    
    for ticker in tickers:
        df = load_from_db(ticker)
        
        needs_fetch = False
        if df.empty:
            needs_fetch = True
        else:
            last_date = df.index.max().date()
            if last_date < (datetime.now().date() - timedelta(days=5)):
                needs_fetch = True
        
        if needs_fetch:
            st.toast(f"Downloading data for {ticker}...", icon="⬇️")
            # auto_adjust=True returns 'Close' which is already adjusted
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not data.empty:
                # Handle MultiIndex columns (Price, Ticker)
                try:
                    # Check if 'Close' is in the top level of columns
                    if 'Close' in data.columns:
                        close_data = data['Close']
                    elif 'Adj Close' in data.columns:
                        close_data = data['Adj Close']
                    else:
                        # Fallback: take the first column if structure is unexpected
                        close_data = data.iloc[:, 0]

                    # If close_data is a DataFrame (e.g. has ticker as column), extract the series
                    if isinstance(close_data, pd.DataFrame):
                        if ticker in close_data.columns:
                            adj_close = close_data[ticker]
                        else:
                            # If ticker name doesn't match exactly or is missing, take first col
                            adj_close = close_data.iloc[:, 0]
                    else:
                        adj_close = close_data
                    
                    clean_df = pd.DataFrame(adj_close)
                    clean_df.columns = ['Adj Close'] # Normalize column name for DB
                    save_to_db(clean_df, ticker)
                    df = clean_df
                except Exception as e:
                    st.error(f"Error processing data for {ticker}: {e}")
                    continue
        
        if not df.empty:
            # DB returns 'adj_close', fresh fetch returns 'Adj Close'
            col_name = 'adj_close' if 'adj_close' in df.columns else 'Adj Close'
            combined_data[ticker] = df[col_name]
    
    return combined_data.dropna()

def calculate_metrics(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def perform_monte_carlo(data, num_simulations=10000, risk_free_rate=0.05):
    log_returns = np.log(data / data.shift(1))
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    num_assets = len(data.columns)
    results = np.zeros((3, num_simulations))
    weights_record = []
    
    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_std = calculate_metrics(weights, mean_returns, cov_matrix)
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std
        
    results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])
    
    max_sharpe_idx = results_df['Sharpe'].idxmax()
    optimal_portfolio = results_df.iloc[max_sharpe_idx]
    optimal_weights = weights_record[max_sharpe_idx]
    
    return results_df, optimal_portfolio, optimal_weights, log_returns

def calculate_var(returns, confidence_level=0.95, investment=100000):
    return 0

def main():
    st.title("📊 Quantitative Portfolio Optimization Engine")
    st.markdown("### Efficient Frontier & Risk Analytics")
    
    st.sidebar.header("User Inputs")
    default_tickers = "RELIANCE.NS, TATASTEEL.NS, GRASIM.NS, HDFCBANK.NS"
    tickers_input = st.sidebar.text_area("Enter Stock Tickers (comma separated)", value=default_tickers)
    tickers = [t.strip() for t in tickers_input.split(',')]
    
    investment_amt = st.sidebar.number_input("Initial Investment (INR)", value=100000, step=1000)
    
    if st.sidebar.button("Run Optimization"):
        start_date = datetime.now() - timedelta(days=5*365)
        end_date = datetime.now()
        
        with st.spinner("Fetching data and running Monte Carlo Simulation..."):
            data = get_data(tickers, start_date, end_date)
            
            if data.empty:
                st.error("No data found. Please check tickers.")
                return
            
            st.success(f"Data loaded for {len(tickers)} assets.")
            
            results_df, opt_port, opt_weights, log_returns = perform_monte_carlo(data)
            
            daily_vol = opt_port['Volatility'] / np.sqrt(252)
            z_score = 1.645
            var_percent = daily_vol * z_score
            var_value = investment_amt * var_percent
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("Efficient Frontier (Monte Carlo)")
                fig = px.scatter(
                    results_df, x='Volatility', y='Return', color='Sharpe',
                    title="10,000 Simulated Portfolios",
                    labels={'Volatility': 'Risk (Volatility)', 'Return': 'Expected Annual Return'},
                    color_continuous_scale='Viridis'
                )
                fig.add_trace(go.Scatter(
                    x=[opt_port['Volatility']], y=[opt_port['Return']],
                    mode='markers', marker=dict(symbol='star', size=15, color='red'),
                    name='Optimal Portfolio'
                ))
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("Optimal Allocation")
                st.metric("Expected Return", f"{opt_port['Return']:.2%}")
                st.metric("Annual Volatility", f"{opt_port['Volatility']:.2%}")
                st.metric("Sharpe Ratio", f"{opt_port['Sharpe']:.2f}")
                st.divider()
                st.metric("Value at Risk (95%)", f"₹{var_value:,.2f}", help="Max expected loss in 1 day with 95% confidence")
                
                weights_df = pd.DataFrame({'Ticker': tickers, 'Weight': opt_weights})
                fig_pie = px.pie(weights_df, values='Weight', names='Ticker', title='Asset Allocation')
                st.plotly_chart(fig_pie, use_container_width=True)

            with st.expander("View Raw Data (First 5 rows)"):
                st.dataframe(data.head())

if __name__ == "__main__":
    main()
