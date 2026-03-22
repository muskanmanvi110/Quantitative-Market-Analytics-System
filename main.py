import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy import log, sqrt, exp
import yfinance as yf
from scipy.optimize import brentq
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
            model="llama-3.1-8b-instant",
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
# Helper Function to Adjust yfinance Data Format
# ------------------------------
def adjust_yf_data(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

# ------------------------------
# Tabs: Unified Analysis
# ------------------------------
tabs = st.tabs(["Option Pricing", "GARCH", "Integrated Analysis & AI"])
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
# Tab 2: GARCH
# ==============================
with tabs[1]:
    st.header("GARCH Model")
    st.write("Forecast volatility on historical returns using a GARCH(1,1) model.")
    ticker_garch = st.text_input("Ticker Symbol for GARCH", value="AAPL", max_chars=10, key="ticker_garch").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))
    data = yf.download(ticker_garch, start=start_date, end=end_date, auto_adjust=False)
    data = adjust_yf_data(data)
    if data.empty:
        st.error("No historical data found for ticker.")
    else:
        data['Return'] = data['Adj Close'].pct_change()
        data = data.dropna()
        fig_returns = px.line(data, x=data.index, y='Return', title=f"{ticker_garch} Daily Returns")
        st.plotly_chart(fig_returns, use_container_width=True)
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


# ==============================
# Tab 3: Integrated Analysis & AI Explanations
# ==============================
with tabs[2]:
    st.header("Integrated Analysis & AI Explanations")
    st.write("Unified analysis with AI-powered explanations for all visualizations and data.")

    # --- FIX: Dynamic ticker input instead of hardcoded "AAPL" ---
    ticker_sym = st.text_input("Ticker Symbol for Analysis", value="AAPL", max_chars=10, key="ticker_integrated").upper()

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
