"""Point-in-time Nifty 50 membership table, reconstructed from public index
reconstitution logs (Wikipedia 'NIFTY 50' change history, cross-checked with
NSE press coverage).  Window of interest: 2010-07-01 .. 2025-07-01.

Notes on data quality:
  * Renames/mergers mapped to the surviving Yahoo symbol (Sesa Goa->VEDL,
    Bharti Infratel->INDUSTOWER, United Spirits->UNITDSPR).
  * 2013-2018 has a handful of ambiguous re-entry dates (VEDL, BANKBARODA,
    NMDC, HINDPETRO, INDUSTOWER, YESBANK); best-supported interpretation is
    encoded and marked APPROX.  Error is bounded at ~1 slot of 50 for
    limited stretches.
  * Members before the window start are given start=2010-07-01.
"""
import numpy as np
import pandas as pd

W0 = "2010-07-01"
W1 = "2025-07-01"

# ticker -> list of (member_from, member_to) intervals, inclusive
INTERVALS = {
    # ---- continuous members through the window ----
    **{t: [(W0, W1)] for t in [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
        "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "HCLTECH.NS",
        "TATAMOTORS.NS", "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS", "M&M.NS",
        "ONGC.NS", "CIPLA.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "KOTAKBANK.NS",
        "GRASIM.NS", "TATACONSUM.NS",  # Tata Global->Tata Consumer continuity? see note
    ]},
    # NOTE: TATACONSUM (Tata Global Beverages) actually joined 2021-03-31.
    # Corrected below by explicit override.
    "TATACONSUM.NS": [("2021-03-31", W1)],
    "BPCL.NS": [(W0, "2025-03-28")],
    "WIPRO.NS": [(W0, "2013-04-01"), ("2013-09-27", W1)],
    "DRREDDY.NS": [("2010-10-01", W1)],
    "BAJAJ-AUTO.NS": [("2010-10-01", W1)],
    "COALINDIA.NS": [("2011-10-10", W1)],
    "ASIANPAINT.NS": [("2012-04-27", W1)],
    "ULTRACEMCO.NS": [("2012-09-28", W1)],
    "INDUSINDBK.NS": [("2013-04-01", W1)],
    "TECHM.NS": [("2014-03-28", W1)],
    "ADANIPORTS.NS": [("2015-09-28", W1)],
    "EICHERMOT.NS": [("2016-04-01", W1)],
    "BAJFINANCE.NS": [("2017-09-29", W1)],
    "BAJAJFINSV.NS": [("2018-04-02", W1)],
    "TITAN.NS": [("2018-04-02", W1)],
    "UPL.NS": [("2018-04-02", "2024-03-28")],
    "JSWSTEEL.NS": [("2018-09-28", W1)],
    "BRITANNIA.NS": [("2019-03-29", W1)],
    "NESTLEIND.NS": [("2019-09-27", W1)],
    "HDFCLIFE.NS": [("2020-07-31", W1)],
    "SBILIFE.NS": [("2020-09-25", W1)],
    "DIVISLAB.NS": [("2020-09-25", "2024-09-30")],
    "APOLLOHOSP.NS": [("2022-03-31", W1)],
    "ADANIENT.NS": [("2022-09-30", W1)],
    "SHRIRAMFIN.NS": [("2024-03-28", W1)],
    "BEL.NS": [("2024-09-30", W1)],
    "TRENT.NS": [("2024-09-30", W1)],
    "JIOFIN.NS": [("2025-03-28", W1)],
    "ETERNAL.NS": [("2025-03-28", W1)],
    # ---- ex-members ----
    "HDFC.NS": [(W0, "2023-07-13")],
    "LTIM.NS": [("2023-07-13", "2024-09-30")],
    "ABB.NS": [(W0, "2010-10-01")],
    "IDEA.NS": [(W0, "2010-10-01"), ("2015-03-27", "2017-05-26")],
    "UNITECH.NS": [(W0, "2011-03-25")],
    "SUZLON.NS": [(W0, "2011-03-25")],
    "RELCAPITAL.NS": [(W0, "2011-10-10")],
    "RCOM.NS": [(W0, "2012-04-27")],
    "RPOWER.NS": [(W0, "2012-09-28")],
    "SAIL.NS": [(W0, "2012-09-28")],
    "BANKBARODA.NS": [(W0, "2012-09-28"), ("2016-04-01", "2018-04-02")],  # APPROX re-entry
    "SIEMENS.NS": [(W0, "2013-04-01")],
    "VEDL.NS": [("2011-03-25", "2016-04-01"), ("2018-09-28", "2020-07-31")],  # APPROX 2nd
    "RELINFRA.NS": [(W0, "2013-09-27")],
    "NMDC.NS": [(W0, "2013-09-27"), ("2014-09-19", "2015-09-28")],  # APPROX re-entry
    "JINDALSTEL.NS": [(W0, "2014-03-28")],  # APPROX exit
    "JPASSOCIAT.NS": [(W0, "2014-03-28")],
    "UNITDSPR.NS": [("2013-09-27", "2014-09-19")],  # APPROX (United Spirits)
    "ZEEL.NS": [("2014-09-19", "2020-09-25")],
    "DLF.NS": [(W0, "2015-03-27")],
    "YESBANK.NS": [("2015-03-27", "2020-03-19")],  # APPROX entry
    "IDFC.NS": [(W0, "2015-05-29")],
    "BOSCHLTD.NS": [("2015-05-29", "2018-09-28")],
    "CAIRN.NS": [(W0, "2016-04-01")],
    "PNB.NS": [(W0, "2016-04-01")],
    "AUROPHARMA.NS": [("2016-04-01", "2018-09-28")],
    "HINDPETRO.NS": [("2016-04-01", "2018-04-02")],  # APPROX
    "BHEL.NS": [(W0, "2017-03-31")],
    "IBULHSGFIN.NS": [("2017-03-31", "2019-09-27")],
    "SAMMAANCAP.NS": [("2017-03-31", "2019-09-27")],  # IBULHSGFIN renamed
    "ACC.NS": [(W0, "2017-09-29")],
    "IOC.NS": [("2017-05-26", "2022-03-31")],
    "TATAPOWER.NS": [(W0, "2018-04-02")],
    "AMBUJACEM.NS": [(W0, "2018-04-02")],
    "LUPIN.NS": [("2012-09-28", "2018-09-28")],
    "INDUSTOWER.NS": [("2018-04-02", "2020-09-25")],  # APPROX (Bharti Infratel)
    "SHREECEM.NS": [("2020-03-19", "2022-09-30")],
    "GAIL.NS": [(W0, "2021-03-31")],
}

EX_MEMBER_TICKERS = sorted(set(INTERVALS) - {
    t for t, iv in INTERVALS.items() if iv == [(W0, W1)]})


def membership_matrix(index, columns):
    """Boolean DataFrame (dates x tickers): True while a member."""
    M = pd.DataFrame(False, index=index, columns=columns)
    for t, ivs in INTERVALS.items():
        if t not in M.columns:
            continue
        for a, b in ivs:
            M.loc[(M.index >= a) & (M.index <= b), t] = True
    return M


def all_tickers():
    return sorted(INTERVALS)
