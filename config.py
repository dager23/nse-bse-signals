"""Global configuration for the NSE strategy research project."""

# ---------------------------------------------------------------- universes
# Nifty 50 constituents (as of mid-2025, best effort; WIPRO deduped)
STOCK_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS",
    "M&M.NS", "ONGC.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "TECHM.NS", "INDUSINDBK.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS",
    "BAJAJFINSV.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS", "EICHERMOT.NS",
    "COALINDIA.NS", "GRASIM.NS", "BPCL.NS", "TATACONSUM.NS", "HINDALCO.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "UPL.NS", "BAJAJ-AUTO.NS", "SHRIRAMFIN.NS",
]

# Top Nifty Midcap 100 names with long listing history (best effort)
MIDCAP_UNIVERSE = [
    "PERSISTENT.NS", "CUMMINSIND.NS", "ASHOKLEY.NS", "AUROPHARMA.NS", "BHARATFORG.NS",
    "TVSMOTOR.NS", "GODREJPROP.NS", "LUPIN.NS", "PIIND.NS", "MPHASIS.NS",
    "COFORGE.NS", "ASTRAL.NS", "POLYCAB.NS", "PAGEIND.NS", "VOLTAS.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "INDHOTEL.NS", "JUBLFOOD.NS", "LTTS.NS",
    "OBEROIRLTY.NS", "MRF.NS", "BALKRISIND.NS", "ESCORTS.NS", "CONCOR.NS",
    "PETRONET.NS", "HINDPETRO.NS", "TATACOMM.NS", "APOLLOTYRE.NS", "SRF.NS",
]

# Top Nifty Smallcap 100 names (best effort; several have shorter history)
SMALLCAP_UNIVERSE = [
    "CDSL.NS", "RADICO.NS", "REDINGTON.NS", "BSOFT.NS", "CYIENT.NS",
    "GRANULES.NS", "LAURUSLABS.NS", "KEC.NS", "NCC.NS", "RBLBANK.NS",
    "IEX.NS", "CESC.NS", "KARURVYSYA.NS", "CANFINHOME.NS", "EXIDEIND.NS",
    "CAMS.NS", "PNBHOUSING.NS", "JBCHEPHARM.NS", "BLUESTARCO.NS", "GESHIP.NS",
]

BENCHMARK = "^NSEI"        # Nifty 50 index
INDIA_VIX = "^INDIAVIX"    # India VIX for volatility strategies

# ---------------------------------------------------------------- costs
TRANSACTION_COST = 0.001     # 0.1% per trade (brokerage + STT + charges)
SLIPPAGE = 0.0005            # 0.05% slippage
INITIAL_CAPITAL = 1_000_000  # Rs 10 Lakh
RISK_FREE_RATE = 0.065       # ~6.5% Indian 10Y govt bond yield

# ---------------------------------------------------------------- windows
TIMEFRAMES = {
    "1Y":  {"start": "2024-07-01", "end": "2025-07-01"},
    "3Y":  {"start": "2022-07-01", "end": "2025-07-01"},
    "5Y":  {"start": "2020-07-01", "end": "2025-07-01"},
    "10Y": {"start": "2015-07-01", "end": "2025-07-01"},
    "15Y": {"start": "2010-07-01", "end": "2025-07-01"},
}
DATA_START = "2009-01-01"    # extra head-room so indicators warm up before 2010
DATA_END = "2025-07-01"

DATA_INTERVALS = ["1d", "1wk", "1mo"]

import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(ROOT, "reports")
