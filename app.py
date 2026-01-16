import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from main import detect_date_col_or_create, detect_revenue_col_or_create, get_dimension_cols, KPIEngine


st.set_page_config(page_title="KPI Agent", layout="wide")

st.title("KPI Intelligence & Action Agent")
st.caption("Dashboard")

with st.sidebar:
    st.header("Data")
    data_file=st.text_input("Dataset path",value="kpi_data.csv")
    rolling_window=st.number_input("Rolling window (days)",min_value=2,max_value=30, value=7)
    threshold=st.number_input("Deviation threshold",min_value=0.01, max_value=1.00, value=0.15, step=0.01)
    days_back=st.number_input("Monitoring horizon (days)", min_value=7,max_value=120, value=30)

@st.cache_data(show_spinner=False)
def load_data(path: str):
    df=pd.read_csv(path)
    df, date_col,date_note=detect_date_col_or_create(df, force_days=90)
    df, rev_col, rev_note=detect_revenue_col_or_create(df)
    dims=get_dimension_cols(df, date_col)
    eng=KPIEngine(df, date_col=date_col,default_kpi=rev_col, dims=dims)
    return eng, date_note,rev_note

try:
    engine, date_note, rev_note=load_data(data_file)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.info(f"{date_note}\n{rev_note}")

with st.sidebar:
    st.header("Filters")
    kpi=st.selectbox("KPI",engine.numeric_cols,index=engine.numeric_cols.index(engine.kpi))
    engine.set_kpi(kpi)

    # filters
    filters={}
    for dim in ["Category","Sub_category","Product_Name"]:
        if dim in engine.df.columns:
            vals=["All"] + sorted(engine.df[dim].dropna().astype(str).unique().tolist())
            choice=st.selectbox(dim, vals, index=0)
            if choice!="All":
                filters[dim]=choice

    st.header("Date range")
    min_dt=engine.df[engine.date_col].min().date()
    max_dt=engine.df[engine.date_col].max().date()
    start_dt=st.date_input("Start", value=max_dt-timedelta(days=int(days_back)-1),min_value=min_dt, max_value=max_dt)
    end_dt=st.date_input("End", value=max_dt, min_value=min_dt, max_value=max_dt)

# Trend
tr=engine.trend(days=int(days_back),filters=filters, start=str(start_dt), end=str(end_dt))
if tr.empty:
    st.warning("No data for the selected slice.")
    st.stop()

col1,col2=st.columns([2,1],gap="large")

with col1:
    st.subheader("Trend")
    fig=plt.figure()
    x=pd.to_datetime(tr[engine.date_col])
    y=tr[engine.kpi].astype(float)
    plt.plot(x, y)
    plt.xticks(rotation=45,ha="right")
    plt.ylabel(engine.kpi)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Latest vs Baseline")
    last_date=pd.to_datetime(tr[engine.date_col]).max()
    last_val=float(tr[pd.to_datetime(tr[engine.date_col])==last_date][engine.kpi].iloc[0])

    _, _, baseline=engine.current_vs_baseline(filters, rolling_window=int(rolling_window), start=str(start_dt), end=str(end_dt))
    if baseline is not None and baseline!=0:
        dev=(last_val - baseline) / baseline
        st.metric("Latest", f"{last_val:,.2f}", delta=f"{dev*100:.1f}% vs baseline")
        st.caption(f"Baseline={baseline:,.2f} (prev {rolling_window}-day rolling avg)")
    else:
        st.metric("Latest",f"{last_val:,.2f}")
        st.caption("Insufficient prior data for baseline in selected range.")

st.subheader("Deviation monitoring")
devs=engine.detect_deviations(days_back=int(days_back), rolling_window=int(rolling_window),threshold=float(threshold), filters=filters, start=str(start_dt),end=str(end_dt))
if not devs:
    st.success("No deviations detected.")
else:
    dev_df=pd.DataFrame([{
        "date": d.date.date(),
        "kpi": d.kpi,
        "actual": d.actual,
        "baseline": d.baseline,
        "deviation_pct": d.deviation_pct,
        "direction": d.direction
    } for d in devs])
    st.dataframe(dev_df, use_container_width=True)

    st.subheader("Causal analysis & recommendations (latest deviation)")
    latest=devs[-1]
    drivers=engine.causal_analysis(str(latest.date.date()),rolling_window=int(rolling_window), filters=filters, top_n=8)
    st.write(f"**{latest.kpi} {latest.direction} {latest.deviation_pct*100:.1f}% on {latest.date.date()}**")
    st.dataframe(drivers[["driver_type","driver","delta","pct_change","z_score","explain"]],use_container_width=True)
    st.caption("Drivers are ranked by abnormal change (z-score when possible,else % change / contribution delta).")
