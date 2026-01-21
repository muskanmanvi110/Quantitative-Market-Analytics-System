import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy import log, sqrt, exp
import yfinance as yf
from datetime import timedelta
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from groq import Groq
from dotenv import load_dotenv
import os

# ------------------------------
# Page Configuration & Custom CSS
# ------------------------------
st.set_page_config(
    page_title="US Options Pricing & Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    body { font-family: 'Segoe UI', sans-serif; }
    .title { text-align: center; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; font-size: 1.25rem; margin-bottom: 2rem; color: #555; }
    .metric-box {
         border-radius: 10px;
         padding: 15px;
         text-align: center;
         color: #fff;
         font-size: 1.5rem;
         font-weight: bold;
         margin: 10px;
    }
    .call-box { background-color: #2ecc71; }
    .put-box { background-color: #e74c3c; }
    .highlight-box {
         background-color: #f8f9fa;
         border-left: 3px solid #4CAF50;
         padding: 15px;
         margin: 10px 0;
         border-radius: 5px;
    }
    .strategy-good { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; }
    .strategy-bad { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
    .strategy-neutral { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">US Options Pricing & Market Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Integrating Black-Scholes, Monte Carlo, Market Data, GARCH, Time Series Forecasting & AI Explanations</div>', unsafe_allow_html=True)

# ------------------------------
# AI Explanation System
# ------------------------------
class FinancialAI:
    def explain(self, description: str) -> str:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        messages = [
            {"role": "system", "content": "You are an expert in financial analysis."},
            {"role": "user", "content": f"Explain this in simple terms for non-technical users: {description}"}
        ]
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated model name
            messages=messages,
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None,
        )
        explanation = ""
        for chunk in completion:
            explanation += chunk.choices[0].delta.content or ""
        return explanation

ai_explainer = FinancialAI()

# ------------------------------
# Black-Scholes Model Definition
# ------------------------------
class BlackScholes:
    def __init__(self, time_to_maturity: float, strike: float, current_price: float, volatility: float, interest_rate: float):
        self.t = time_to_maturity
        self.K = strike
        self.S = current_price
        self.sigma = volatility
        self.r = interest_rate

    def calculate_prices(self):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.t) / (self.sigma * sqrt(self.t))
        d2 = d1 - self.sigma * sqrt(self.t)
        call_price = self.S * norm.cdf(d1) - self.K * exp(-self.r * self.t) * norm.cdf(d2)
        put_price = self.K * exp(-self.r * self.t) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return call_price, put_price

# ------------------------------
# Helper Functions for Visuals
# ------------------------------
def generate_heatmap_data(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            temp_model = BlackScholes(bs_model.t, strike, spot, vol, bs_model.r)
            cp, pp = temp_model.calculate_prices()
            call_prices[i, j] = cp
            put_prices[i, j] = pp
    return call_prices, put_prices

def create_plotly_heatmap(data, x, y, title):
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=np.round(x, 2),
        y=np.round(y, 2),
        colorscale='Viridis',
        colorbar=dict(title="Price (USD)")
    ))
    fig.update_layout(title=title, xaxis_title="Spot Price (USD)", yaxis_title="Volatility")
    return fig

def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan
    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price
    try:
        vol = brentq(objective, 1e-6, 5)
    except Exception:
        vol = np.nan
    return vol

# ------------------------------
# Monte Carlo Simulation & VaR Calculation
# ------------------------------
def monte_carlo_option_price(S, K, T, r, sigma, n_sim=10000, n_steps=100):
    dt = T / n_steps
    rand = np.random.normal(size=(n_steps, n_sim))
    price_paths = np.zeros((n_steps+1, n_sim))
    price_paths[0] = S
    for t in range(1, n_steps+1):
        price_paths[t] = price_paths[t-1] * np.exp((r - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * rand[t-1])
    payoffs = np.maximum(price_paths[-1] - K, 0)
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    return mc_price, price_paths

def calculate_var(price_paths, confidence=95):
    final_prices = price_paths[-1]
    returns = (final_prices / price_paths[0]) - 1
    var = np.percentile(returns, 100 - confidence)
    return var

# ------------------------------
# Performance Metrics for Backtest
# ------------------------------
def calculate_performance_metrics(strategy_returns, market_returns):
    """Calculate key performance metrics for a strategy vs. benchmark"""
    total_strategy_return = (strategy_returns.iloc[-1] / strategy_returns.iloc[0]) - 1
    total_market_return = (market_returns.iloc[-1] / market_returns.iloc[0]) - 1
    n_years = len(strategy_returns) / 252
    ann_strategy_return = (1 + total_strategy_return) ** (1 / n_years) - 1
    ann_market_return = (1 + total_market_return) ** (1 / n_years) - 1
    strategy_vol = np.std(strategy_returns.pct_change().dropna()) * np.sqrt(252)
    market_vol = np.std(market_returns.pct_change().dropna()) * np.sqrt(252)
    strategy_sharpe = ann_strategy_return / strategy_vol if strategy_vol > 0 else 0
    market_sharpe = ann_market_return / market_vol if market_vol > 0 else 0
    strategy_dd = (strategy_returns / strategy_returns.cummax() - 1).min()
    market_dd = (market_returns / market_returns.cummax() - 1).min()
    return {
        "Total Return": [total_strategy_return, total_market_return],
        "Annualized Return": [ann_strategy_return, ann_market_return],
        "Annualized Volatility": [strategy_vol, market_vol],
        "Sharpe Ratio": [strategy_sharpe, market_sharpe],
        "Maximum Drawdown": [strategy_dd, market_dd]
    }

# ------------------------------
# Helper Function to Adjust yfinance Data Format
# ------------------------------
def adjust_yf_data(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

# ------------------------------
# Tabs: Unified Analysis
# ------------------------------
tabs = st.tabs(["Option Pricing", "Market Analysis", "GARCH & Backtesting", "Integrated Analysis & AI", "Mean Reversion Strategy"])
current_date = pd.Timestamp.today().normalize()

# ==============================
# Tab 1: Option Pricing (US Context)
# ==============================
with tabs[0]:
    st.header("Black-Scholes Option Pricing")
    st.write("Enter parameters in US Dollars (USD) to calculate theoretical option prices for US stocks.")
    col_params = st.columns(5)
    with col_params[0]:
        S = st.number_input("Asset Price (S in USD)", value=150.0, min_value=1.0)
    with col_params[1]:
        K = st.number_input("Strike Price (K in USD)", value=150.0, min_value=1.0)
    with col_params[2]:
        t = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.1)
    with col_params[3]:
        sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=1.0, step=0.01)
    with col_params[4]:
        r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, step=0.01)
    bs = BlackScholes(t, K, S, sigma, r)
    call_val, put_val = bs.calculate_prices()
    col_prices = st.columns(2)
    with col_prices[0]:
        st.markdown(f'<div class="metric-box call-box">CALL: ${call_val:.2f}</div>', unsafe_allow_html=True)
    with col_prices[1]:
        st.markdown(f'<div class="metric-box put-box">PUT: ${put_val:.2f}</div>', unsafe_allow_html=True)
    if st.button("Explain Option Prices", key="explain_bs"):
        explanation = ai_explainer.explain(f"""
        The Black-Scholes model has calculated that:
        - A call option with strike price ${K} expiring in {t:.2f} years is worth ${call_val:.2f}
        - A put option with the same parameters is worth ${put_val:.2f}
        The underlying asset price is ${S}, volatility is {sigma*100:.1f}%, and the risk-free rate is {r*100:.1f}%.
        """)
        st.write(explanation)
    st.subheader("Interactive Heatmaps")
    col_heat = st.columns(2)
    with col_heat[0]:
        spot_min = st.number_input("Min Spot Price (USD)", value=S*0.8, min_value=0.1, step=0.1, key="min_spot_us")
    with col_heat[1]:
        spot_max = st.number_input("Max Spot Price (USD)", value=S*1.2, min_value=0.1, step=0.1, key="max_spot_us")
    vol_min = st.slider("Min Volatility", min_value=0.01, max_value=1.0, value=sigma*0.5, step=0.01)
    vol_max = st.slider("Max Volatility", min_value=0.01, max_value=1.0, value=sigma*1.5, step=0.01)
    spot_range = np.linspace(spot_min, spot_max, 20)
    vol_range = np.linspace(vol_min, vol_max, 20)
    call_prices, put_prices = generate_heatmap_data(bs, spot_range, vol_range, K)
    fig_call = create_plotly_heatmap(call_prices, spot_range, vol_range, "Call Price Heatmap")
    fig_put = create_plotly_heatmap(put_prices, spot_range, vol_range, "Put Price Heatmap")
    col_maps = st.columns(2)
    with col_maps[0]:
        st.plotly_chart(fig_call, use_container_width=True)
    with col_maps[1]:
        st.plotly_chart(fig_put, use_container_width=True)
    if st.button("Explain Heatmaps", key="explain_heatmaps"):
        explanation = ai_explainer.explain(f"""
        These heatmaps show how call and put option prices change when:
        1. The stock price moves (horizontal axis)
        2. The volatility changes (vertical axis)
        For call options (left), prices increase (brighter colors) when stock prices rise or when volatility increases.
        For put options (right), prices increase when stock prices fall or when volatility increases.
        This visualization helps traders understand option price sensitivity to market conditions.
        """)
        st.write(explanation)
    prices_df = pd.DataFrame({
        "Call Price (USD)": [call_val],
        "Put Price (USD)": [put_val]
    })
    st.download_button("Download Option Prices", prices_df.to_csv(index=False), "option_prices.csv", "text/csv")

# ==============================
# Tab 2: Market Analysis (US Context)
# ==============================
with tabs[1]:
    st.header("Market Analysis: Implied Volatility & Historical Data")
    st.write("Retrieve live market data for US stocks to compute the implied volatility surface and display historical price trends.")
    col_market = st.columns(3)
    with col_market[0]:
        ticker_sym = st.text_input("Ticker Symbol", value="AAPL", max_chars=10, key="ticker_market").upper()
    with col_market[1]:
        market_r = st.number_input("Risk-Free Rate", value=0.05, step=0.005, format="%.4f")
    with col_market[2]:
        div_yield = st.number_input("Dividend Yield", value=0.02, step=0.005, format="%.4f")
    ticker_obj = yf.Ticker(ticker_sym)
    try:
        expirations = ticker_obj.options
    except Exception as e:
        st.error(f"Error fetching options for {ticker_sym}: {e}. Try another ticker (e.g., 'MSFT', 'GOOG').")
        st.stop()
    if not expirations:
        st.error(f"No options data available for {ticker_sym}.")
        st.stop()
    exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > current_date + timedelta(days=7)]
    if not exp_dates:
        st.error(f"No valid expiration dates found for {ticker_sym}.")
    else:
        options_list = []
        for expiration_date in exp_dates:
            try:
                chain = ticker_obj.option_chain(expiration_date.strftime('%Y-%m-%d'))
                calls = chain.calls
            except Exception:
                continue
            calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
            for _, row in calls.iterrows():
                mid_price = (row['bid'] + row['ask']) / 2
                options_list.append({
                    "expiration": expiration_date,
                    "strike": row["strike"],
                    "mid": mid_price
                })
        if not options_list:
            st.error("No option data available after filtering.")
        else:
            options_df = pd.DataFrame(options_list)
            try:
                hist = ticker_obj.history(period="5d", auto_adjust=False)
                hist = adjust_yf_data(hist)
                if hist.empty:
                    st.error("No historical data available.")
                    st.stop()
                else:
                    spot_price = hist["Close"].iloc[-1]
            except Exception as e:
                st.error(f"Error fetching historical data: {e}")
                st.stop()
            options_df["daysToExp"] = (options_df["expiration"] - current_date).dt.days
            options_df["T"] = options_df["daysToExp"] / 365
            options_df["ImplVol"] = options_df.apply(
                lambda row: implied_volatility(row["mid"], spot_price, row["strike"], row["T"], market_r, div_yield),
                axis=1
            )
            options_df = options_df.dropna(subset=["ImplVol"])
            options_df["ImplVol"] *= 100  # Convert to percentage
            X = options_df["T"].values
            Y_vals = options_df["strike"].values
            Z = options_df["ImplVol"].values
            ti = np.linspace(X.min(), X.max(), 50)
            yi = np.linspace(Y_vals.min(), Y_vals.max(), 50)
            T_mesh, Y_mesh = np.meshgrid(ti, yi)
            Zi = griddata((X, Y_vals), Z, (T_mesh, Y_mesh), method="linear")
            Zi = np.ma.array(Zi, mask=np.isnan(Zi))
            fig_surface = go.Figure(data=[go.Surface(
                x=T_mesh, y=Y_mesh, z=Zi,
                colorscale='Viridis',
                colorbar=dict(title="Impl. Vol (%)")
            )])
            fig_surface.update_layout(
                title=f"Implied Volatility Surface for {ticker_sym}",
                scene=dict(
                    xaxis_title="Time to Expiration (years)",
                    yaxis_title="Strike Price (USD)",
                    zaxis_title="Implied Volatility (%)"
                ),
                autosize=True,
                margin=dict(l=65, r=50, b=65, t=90)
            )
            st.plotly_chart(fig_surface, use_container_width=True)
            if st.button("Explain Volatility Surface", key="explain_vol_surface"):
                explanation = ai_explainer.explain(f"""
                The 3D volatility surface for {ticker_sym} shows how the market prices risk:
                - The x-axis shows time to expiration (longer-dated options are further away)
                - The y-axis shows different strike prices (higher strikes to the right)
                - The z-axis (height and color) shows implied volatility percentage
                The shape of this surface reveals market expectations:
                - Slopes indicate expected directional movement
                - "Smiles" or "smirks" (curves) show tail risk concerns
                - Higher implied volatility means the market expects larger price movements
                This is the market's forecast of future volatility based on real option prices.
                """)
                st.write(explanation)
            st.subheader(f"Historical Prices for {ticker_sym}")
            fig_hist = px.line(hist, x=hist.index, y="Close", title=f"{ticker_sym} Historical Close Prices",
                               labels={"Close": "Price (USD)", "Date": "Date"})
            st.plotly_chart(fig_hist, use_container_width=True)
            st.download_button("Download Options Data", options_df.to_csv(index=False), "market_options.csv", "text/csv")

# ==============================
# Tab 3: GARCH & Backtesting
# ==============================
with tabs[2]:
    st.header("GARCH Model & Backtesting")
    st.write("Forecast volatility on historical returns using a GARCH(1,1) model and backtest a simple trading strategy based on volatility forecasts.")
    ticker_garch = st.text_input("Ticker Symbol for GARCH & Backtesting", value="AAPL", max_chars=10, key="ticker_garch").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))
    data = yf.download(ticker_garch, start=start_date, end=end_date, auto_adjust=False)
    data = adjust_yf_data(data)
    if data.empty:
        st.error("No historical data found for ticker.")
    else:
        data['Return'] = data['Adj Close'].pct_change()
        data = data.dropna()
        # Display historical returns and volatility
        fig_returns = px.line(data, x=data.index, y='Return', title=f"{ticker_garch} Daily Returns")
        st.plotly_chart(fig_returns, use_container_width=True)
        # GARCH model fitting
        st.subheader("GARCH(1,1) Model Results")
        with st.spinner("Fitting GARCH model..."):
            am = arch_model(data['Return']*100, vol='Garch', p=1, q=1, dist='Normal')
            res = am.fit(disp="off")
            omega = res.params['omega']
            alpha = res.params['alpha[1]']
            beta = res.params['beta[1]']
            persistence = alpha + beta
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Constant (ω)", f"{omega:.6f}")
            col2.metric("ARCH (α)", f"{alpha:.6f}")
            col3.metric("GARCH (β)", f"{beta:.6f}")
            col4.metric("Persistence (α+β)", f"{persistence:.6f}", 
                        delta_color="off" if 0.95 <= persistence <= 1 else ("normal" if persistence < 0.95 else "inverse"))
            with st.expander("View detailed GARCH model statistics"):
                st.text(res.summary())
        if st.button("Explain GARCH Model Results", key="explain_garch"):
            explanation = ai_explainer.explain(f"""
            The GARCH(1,1) model for {ticker_garch} shows how volatility changes over time:
            - Constant (ω): {omega:.6f} - The baseline volatility when there are no shocks
            - ARCH (α): {alpha:.6f} - How much recent price shocks affect today's volatility
            - GARCH (β): {beta:.6f} - How persistent volatility is over time
            - Persistence (α+β): {persistence:.6f} - How long volatility shocks last
            {"This stock shows high volatility persistence, meaning that when volatility increases, it tends to stay high for extended periods." if persistence > 0.9 else "This stock shows moderate volatility persistence."}
            {"The model indicates volatility is very sensitive to market shocks." if alpha > 0.1 else "The model shows volatility is relatively stable against market shocks."}
            """)
            st.write(explanation)
        # Forecasting
        st.subheader("Conditional Volatility")
        conditional_vol = res.conditional_volatility
        data['GARCH_Vol'] = conditional_vol / 100
        fig_vol = px.line(data, x=data.index, y='GARCH_Vol', 
                           title=f"{ticker_garch} Conditional Volatility (GARCH)")
        fig_vol.update_yaxes(title="Annualized Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)
        if st.button("Explain Volatility Chart", key="explain_vol_chart"):
            avg_vol = data['GARCH_Vol'].mean()
            recent_vol = data['GARCH_Vol'].iloc[-30:].mean()
            vol_trend = "increasing" if data['GARCH_Vol'].iloc[-30:].mean() > data['GARCH_Vol'].iloc[-60:-30].mean() else "decreasing"
            explanation = ai_explainer.explain(f"""
            This chart shows the changing volatility of {ticker_garch} stock over time as estimated by the GARCH model:
            - Peaks represent periods of market stress or uncertainty
            - Valleys show calmer trading periods
            - The average volatility is {avg_vol:.2%} (annualized)
            - Recent volatility has been {recent_vol:.2%}, which is {"higher" if recent_vol > avg_vol else "lower"} than the historical average
            - The trend in volatility is currently {vol_trend}
            Investors can use this information to adjust position sizing, option strategies, or timing of trades.
            {"Higher volatility periods may offer better option-selling opportunities but require smaller position sizes for directional trades." if recent_vol > avg_vol else "Lower volatility periods may be better for building directional positions but offer less premium for option sellers."}
            """)
            st.write(explanation)
        # Strategy Backtest
        st.subheader("Trading Strategy Backtest")
        st.write("This strategy goes long when forecasted volatility is below its median and stays out of the market when volatility is high.")
        window = st.slider("Rolling Window Size (trading days)", min_value=60, max_value=500, value=250, step=10)
        with st.spinner("Running backtest..."):
            forecasts = []
            for i in range(window, len(data)):
                train = data['Return'].iloc[i-window:i] * 100
                model = arch_model(train, vol='Garch', p=1, q=1, dist='Normal')
                model_res = model.fit(disp="off")
                fcast = model_res.forecast(horizon=1)
                vol_forecast = np.sqrt(fcast.variance.values[-1, 0])
                forecasts.append(vol_forecast)
            forecast_index = data.index[window:]
            forecast_series = pd.Series(forecasts, index=forecast_index)
            median_vol = forecast_series.median()
            data['Signal'] = 0
            data.loc[forecast_series.index, 'Signal'] = (forecast_series < median_vol).astype(int)
            data['Strategy Return'] = data['Signal'].shift(1) * data['Return']
            data['Cumulative Strategy'] = (1 + data['Strategy Return']).cumprod()
            data['Cumulative BuyHold'] = (1 + data['Return']).cumprod()
            backtest_data = data.loc[forecast_index]
            fig_backtest = px.line(backtest_data, x=backtest_data.index, 
                                  y=['Cumulative Strategy', 'Cumulative BuyHold'],
                                  labels={'value': 'Cumulative Return', 'variable': 'Strategy'},
                                  title=f"Volatility-Based Strategy vs Buy-and-Hold for {ticker_garch}")
            fig_backtest.update_layout(legend_title_text='Strategy')
            st.plotly_chart(fig_backtest, use_container_width=True)
            metrics = calculate_performance_metrics(
                backtest_data['Cumulative Strategy'], 
                backtest_data['Cumulative BuyHold']
            )
            metrics_df = pd.DataFrame(metrics, index=['Strategy', 'Buy & Hold'])
            formatted_metrics = metrics_df.copy()
            for col in ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown']:
                formatted_metrics[col] = formatted_metrics[col].map('{:.2%}'.format)
            formatted_metrics['Sharpe Ratio'] = formatted_metrics['Sharpe Ratio'].map('{:.2f}'.format)
            st.table(formatted_metrics)
            strategy_return = metrics['Total Return'][0]
            market_return = metrics['Total Return'][1]
            strategy_sharpe = metrics['Sharpe Ratio'][0]
            market_sharpe = metrics['Sharpe Ratio'][1]
            if strategy_return > market_return and strategy_sharpe > market_sharpe:
                performance = "good"
                message = "The volatility-based strategy outperformed the market on both return and risk-adjusted basis."
            elif strategy_return > market_return:
                performance = "neutral"
                message = "The strategy generated higher returns but with higher risk."
            elif strategy_sharpe > market_sharpe:
                performance = "neutral"
                message = "The strategy had better risk-adjusted returns but underperformed on absolute returns."
            else:
                performance = "bad"
                message = "The volatility-based strategy underperformed the market on both metrics."
            st.markdown(f'<div class="strategy-{performance}">{message}</div>', unsafe_allow_html=True)
        if st.button("Explain Strategy & Results", key="explain_strategy"):
            if 'backtest_data' in locals():
                strategy_final = backtest_data['Cumulative Strategy'].iloc[-1]
                market_final = backtest_data['Cumulative BuyHold'].iloc[-1]
                win_rate = (backtest_data['Strategy Return'] > 0).mean()
                max_up = backtest_data['Strategy Return'].max()
                max_down = backtest_data['Strategy Return'].min()
                explanation = ai_explainer.explain(f"""
                This strategy trades {ticker_garch} based on expected volatility:
                - When volatility forecast is LOW (below median), the strategy BUYS the stock
                - When volatility forecast is HIGH (above median), the strategy STAYS OUT of the market
                The logic: Stocks typically offer better risk-adjusted returns during low-volatility periods.
                Results:
                - $10,000 in this strategy would now be worth ${10000*strategy_final:.2f} vs ${10000*market_final:.2f} for buy-and-hold
                - Win rate: {win_rate:.1%} of trading days were profitable
                - Best daily gain: {max_up:.2%}
                - Worst daily loss: {max_down:.2%}
                What this means for investors:
                {"This strategy successfully filtered out high-volatility periods that typically offer poor risk-adjusted returns." if strategy_sharpe > market_sharpe else "This strategy wasn't effective at timing the market based on volatility forecasts for this particular stock."}
                {"Consider using volatility forecasts as part of your risk management system." if strategy_sharpe > market_sharpe else "For this stock, a simple buy-and-hold approach has been more effective than trying to time based on volatility."}
                """)
            else:
                explanation = ai_explainer.explain(f"""
                This strategy would trade {ticker_garch} based on expected volatility:
                - When volatility forecast is LOW (below median), the strategy would BUY the stock
                - When volatility forecast is HIGH (above median), the strategy would STAY OUT of the market
                The logic: Stocks typically offer better risk-adjusted returns during low-volatility periods.
                Please run the backtest to see the specific performance results for this stock.
                """)
            st.write(explanation)
        if 'backtest_data' in locals():
            backtest_download = backtest_data[['Return', 'Signal', 'Strategy Return', 
                                             'Cumulative Strategy', 'Cumulative BuyHold']].copy()
            st.download_button("Download Backtest Results", 
                              backtest_download.to_csv(index=True), 
                              "garch_backtest_results.csv", 
                              "text/csv")

# ==============================
# Tab 4: Integrated Analysis & AI Explanations
# ==============================
with tabs[3]:
    st.header("Integrated Analysis & AI Explanations")
    st.write("Unified analysis with AI-powered explanations for all visualizations and data.")
    
    # Subsection: Option Pricing Comparison
    st.subheader("Option Pricing Comparison")
    bs_call, bs_put = bs.calculate_prices()
    mc_call, mc_paths = monte_carlo_option_price(S, K, t, r, sigma)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Black-Scholes Call", f"${bs_call:.2f}")
    with col2:
        st.metric("Monte Carlo Call", f"${mc_call:.2f}")
    
    if st.button("Explain Pricing Methods", key="explain_pricing_methods"):
        explanation = ai_explainer.explain(
            "The Black-Scholes model calculates option prices using a closed-form solution based on assumptions like constant volatility and no jumps in price. "
            "Monte Carlo simulation, on the other hand, uses random sampling to simulate thousands of possible price paths and averages the results. "
            "While Black-Scholes is faster and simpler, Monte Carlo is more flexible and can handle complex scenarios like path-dependent options."
        )
        st.write(explanation)
    
    # Subsection: Value at Risk (VaR) Calculation Using Monte Carlo
    st.subheader("Value at Risk (VaR) Calculation Using Monte Carlo")
    var_95 = calculate_var(mc_paths, confidence=95)
    var_99 = calculate_var(mc_paths, confidence=99)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("95% VaR", f"{var_95:.4f}")
    with col2:
        st.metric("99% VaR", f"{var_99:.4f}")
    
    if st.button("Explain VaR", key="explain_var"):
        explanation = ai_explainer.explain(
            f"Value at Risk (VaR) measures the maximum potential loss over a specific time period with a given confidence level. "
            f"For this stock ({ticker_sym}) with current price ${S:.2f}, the 95% VaR is {var_95:.4f}, meaning there's a 5% chance of losing more than this amount. "
            f"The 99% VaR is {var_99:.4f}, indicating a 1% chance of exceeding this loss. VaR helps investors understand their risk exposure."
        )
        st.write(explanation)
    
    # Subsection: Time Series Forecasting (ARIMA)
    st.subheader("Time Series Forecasting (ARIMA)")
    try:
        data_ts = yf.download(ticker_sym, period="1y", auto_adjust=False)
        data_ts = adjust_yf_data(data_ts)
        if not data_ts.empty:
            model_arima = ARIMA(data_ts['Adj Close'], order=(1, 1, 1))
            model_fit = model_arima.fit()
            forecast = model_fit.forecast(steps=30)
            
            forecast_dates = pd.date_range(start=data_ts.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
            forecast_df = pd.DataFrame({
                'Forecast': forecast.values
            }, index=forecast_dates)
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=data_ts.index, y=data_ts['Adj Close'],
                mode='lines',
                name='Historical'
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df['Forecast'],
                mode='lines',
                line=dict(dash='dash'),
                name='Forecast'
            ))
            fig_forecast.update_layout(
                title=f"30-Day Price Forecast for {ticker_sym} (ARIMA)",
                xaxis_title="Date",
                yaxis_title="Price (USD)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            if st.button("Explain Forecast", key="explain_forecast"):
                current_price = data_ts['Adj Close'].iloc[-1]
                forecast_price = forecast.values[-1]
                direction = "upward" if forecast_price > current_price else "downward"
                percent_change = (forecast_price / current_price - 1) * 100
                
                explanation = ai_explainer.explain(
                    f"The ARIMA forecast predicts a {direction} trend for {ticker_sym} over the next 30 trading days. "
                    f"The current price is ${current_price:.2f}, and the forecasted price is ${forecast_price:.2f}, representing a {percent_change:.2f}% change. "
                    "This forecast assumes that historical patterns will continue, but unexpected events can alter the trajectory."
                )
                st.write(explanation)
        else:
            st.error("Insufficient data for forecasting.")
    except Exception as e:
        st.error(f"Error in time series forecasting: {e}")
    
    # Subsection: Volatility Clustering Analysis
    st.subheader("Volatility Clustering Analysis")
    try:
        data_ts['Returns'] = data_ts['Adj Close'].pct_change().dropna()
        data_ts['Abs Returns'] = abs(data_ts['Returns'])
        data_ts['Rolling Vol'] = data_ts['Returns'].rolling(window=21).std() * np.sqrt(252)
        
        fig_vol = px.line(data_ts, y='Rolling Vol', title=f"21-Day Rolling Annualized Volatility for {ticker_sym}")
        fig_vol.update_layout(yaxis_title="Annualized Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Plot autocorrelation of absolute returns
        from statsmodels.graphics.tsaplots import plot_acf
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_acf(data_ts['Abs Returns'].dropna(), ax=ax, lags=30)
        plt.title("Autocorrelation of Absolute Returns (Volatility Clustering)")
        st.pyplot(fig)
        
        if st.button("Explain Volatility Analysis", key="explain_vol_analysis"):
            current_vol = data_ts['Rolling Vol'].iloc[-1]
            avg_vol = data_ts['Rolling Vol'].mean()
            max_vol = data_ts['Rolling Vol'].max()
            volatility_regime = "high" if current_vol > avg_vol else "low"
            
            explanation = ai_explainer.explain(
                f"The volatility analysis for {ticker_sym} shows: "
                f"1. Current volatility: {current_vol:.2%}, Average volatility: {avg_vol:.2%}, Maximum volatility: {max_vol:.2%}. "
                f"The stock is currently in a {volatility_regime} volatility regime. "
                "2. The autocorrelation chart confirms 'volatility clustering,' where high-volatility periods tend to persist, and calm periods also cluster together. "
                "This insight is crucial for risk management and strategy development."
            )
            st.write(explanation)
    except Exception as e:
        st.error(f"Error in volatility clustering analysis: {e}")

with tabs[4]:
    st.header("Mean Reversion Strategy")
    st.write("Identify overbought/oversold conditions based on price deviations from recent averages.")
    
    # Input parameters
    col_mr = st.columns(3)
    with col_mr[0]:
        ticker_mr = st.text_input("Ticker Symbol", value="AAPL", max_chars=10, key="ticker_mr").upper()
    with col_mr[1]:
        lookback = st.number_input("Lookback Period (days)", value=5, min_value=2, max_value=30, key="lookback_mr")
    with col_mr[2]:
        threshold = st.number_input("Threshold (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.5, key="threshold_mr")
    
    # Fetch data
    try:
        data_mr = yf.download(ticker_mr, period="1y", auto_adjust=False)
        data_mr = adjust_yf_data(data_mr)
        if data_mr.empty:
            st.error("No historical data found")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()
    
    # Calculate signals
    data_mr['Return'] = data_mr['Close'].pct_change(lookback)
    data_mr['Signal'] = np.where(
        data_mr['Return'] > threshold / 100, -1,
        np.where(data_mr['Return'] < -threshold / 100, 1, 0)
    )
    
    # Plotting
    fig_mr = go.Figure()
    
    # Price line
    fig_mr.add_trace(go.Scatter(
        x=data_mr.index,
        y=data_mr['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#7FFFD4')  # Aquamarine for dark theme
    ))
    
    # Buy signals
    buy_signals = data_mr[data_mr['Signal'] == 1]
    fig_mr.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(
            color='#2ECC71',  # Green
            size=10,
            symbol='triangle-up'
        )
    ))
    
    # Sell signals
    sell_signals = data_mr[data_mr['Signal'] == -1]
    fig_mr.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(
            color='#E74C3C',  # Red
            size=10,
            symbol='triangle-down'
        )
    ))
    
    # Update layout for dark theme
    fig_mr.update_layout(
        title=f"{ticker_mr} Price with Mean Reversion Signals",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display plot
    st.plotly_chart(fig_mr, use_container_width=True)
    
    # AI Explanation
    if st.button("Explain Strategy & Signals", key="explain_mr"):
        explanation = ai_explainer.explain(f"""
        The Mean Reversion Strategy for {ticker_mr} works as follows:
        1. Looks at {lookback}-day price changes
        2. Identifies overbought conditions when price rises more than {threshold}% 
        3. Identifies oversold conditions when price falls more than {threshold}%
        4. Generates sell signals (red triangles) for overbought conditions
        5. Generates buy signals (green triangles) for oversold conditions
        
        Key observations from the current data:
        - {len(buy_signals)} buy signals generated in the last year
        - {len(sell_signals)} sell signals generated in the last year
        - The strategy assumes prices will revert to their average after these extremes
        
        Investors should:
        - Consider selling when seeing red triangles (overbought)
        - Look for buying opportunities at green triangles (oversold)
        - Use this in combination with other indicators for confirmation
        """)
        st.write(explanation)
    
    # Performance summary
    st.subheader("Strategy Performance Summary")
    total_signals = len(buy_signals) + len(sell_signals)
    hit_rate = (len(buy_signals[data_mr.loc[buy_signals.index]['Return'].shift(-1) > 0]) + 
                len(sell_signals[data_mr.loc[sell_signals.index]['Return'].shift(-1) < 0])) / total_signals
    
    col_perf = st.columns(3)
    with col_perf[0]:
        st.metric("Total Signals", total_signals)
    with col_perf[1]:
        st.metric("Hit Rate", f"{hit_rate:.1%}")
    with col_perf[2]:
        st.metric("Last Signal", 
                 "Buy" if data_mr['Signal'].iloc[-1] == 1 else 
                 "Sell" if data_mr['Signal'].iloc[-1] == -1 else "Hold")
    
    # Download data
    st.download_button(
        "Download Signal Data",
        data_mr[['Close', 'Return', 'Signal']].to_csv(),
        file_name=f"{ticker_mr}_mean_reversion_signals.csv"
    )

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>AI-enhanced Financial Analysis</p>
        <p>Developed with Streamlit, Plotly, ARCH, and Groq AI integration</p>
    </div>
""", unsafe_allow_html=True)
