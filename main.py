import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime,timedelta
from typing import Dict,List,Optional,Tuple

import numpy as np
import pandas as pd

 
#Helpers

def normalize_col(c: str)->str:
    return re.sub(r"[^a-z0-9]+","_",str(c).strip().lower())

def safe_to_numeric(s: pd.Series)->pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s,errors="coerce")

def fmt_num(x: float)->str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

def fmt_pct(x: float)->str:
    return f"{x*100:.1f}%"

def print_box(title: str,lines: List[str]):
    print(title)
    for ln in lines:
        print(ln)
    print("")


#Robust Date + Revenue setup

DATE_HINTS = {"date","day","dt","timestamp"}
REVENUE_HINTS = {"overall_revenue","revenue","total_revenue","gross_revenue","net_revenue"}

def detect_date_col_or_create(df: pd.DataFrame, force_days: int = 90)->Tuple[pd.DataFrame, str, str]:
    """
    Returns (df,date_col,setup_note)
    -If a parseable date column exists, use it.
    -Otherwise synthesize a 'Date' column mapped onto a fixed 90-day daily sequence.
    """
    #Direct hint match
    for c in df.columns:
        if normalize_col(c) in DATE_HINTS:
            parsed = pd.to_datetime(df[c],errors="coerce")
            if parsed.notna().mean()>=0.6:
                df[c] = parsed
                return df, c, f"- Detected date column '{c}'."

    #Parseability scan (object-like)
    best = None
    best_score = -1.0
    for c in df.columns:
        if df[c].dtype == "O" or np.issubdtype(df[c].dtype, np.datetime64):
            parsed = pd.to_datetime(df[c], errors="coerce")
            score = float(parsed.notna().mean())
            if score> best_score:
                best_score = score
                best = c
    if best is not None and best_score>=0.6:
        df[best] = pd.to_datetime(df[best], errors="coerce")
        return df, best, f"- Detected date-like column '{best}'."

    #Fallback:synthesize a fixed 90-day daily sequence (deterministic)
    n = len(df)
    days = force_days if n>=force_days else n
    end = datetime.now().date()  # today
    start = end-timedelta(days=days-1)
    dates = pd.date_range(start=start, periods=days, freq="D")

    if n == days:
        df["Date"] = dates
    else:
        df["Date"] = dates[(np.arange(n) % days)]

    note = f"- No explicit date column found. Synthesized 'Date' by mapping row order onto a {days}-day daily sequence ({dates[0].date()} → {dates[-1].date()})."
    return df, "Date", note

def detect_revenue_col_or_create(df: pd.DataFrame)->Tuple[pd.DataFrame, str, str]:
    """
    Returns (df, revenue_col, setup_note)
    If a revenue-like numeric column exists, use it.
    Otherwise derive: Overall_Revenue = Price × Sales_m × (1 − Discount%).
    """
    norm = {c: normalize_col(c) for c in df.columns}
    for c in df.columns:
        if norm[c] in REVENUE_HINTS and pd.api.types.is_numeric_dtype(df[c]):
            return df, c, f"- Using provided revenue column '{c}'."

    needed = {"price", "sales_m", "discount"}
    have = set(norm.values())
    if needed.issubset(have):
        price_col = next(c for c in df.columns if norm[c] == "price")
        sales_col = next(c for c in df.columns if norm[c] == "sales_m")
        disc_col = next(c for c in df.columns if norm[c] == "discount")

        price = safe_to_numeric(df[price_col]).fillna(0)
        sales = safe_to_numeric(df[sales_col]).fillna(0)
        disc = safe_to_numeric(df[disc_col]).fillna(0).clip(0, 100)

        df["Overall_Revenue"] = price * sales * (1-disc / 100.0)
        return df, "Overall_Revenue", "- Derived 'Overall_Revenue' as Price × Sales_m × (1 − Discount%)."

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        c = max(numeric_cols,key=lambda x: float(pd.to_numeric(df[x],errors="coerce").abs().mean()))
        return df, c,f"- Revenue column not found; using '{c}' as KPI proxy."

    raise ValueError("No numeric columns found to build KPI.")

def get_dimension_cols(df: pd.DataFrame,date_col: str)->List[str]:
    dims = []
    for c in df.columns:
        if c == date_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            dims.append(c)
        else:
            if df[c].nunique(dropna=True) <= 25:
                dims.append(c)
    return dims


#Safe CSV logging (no crashes)

def append_row_csv(path: str, row: Dict)->str:
    """
    Append one row to CSV. If locked (Windows/Excel), write to a timestamped file instead.
    Returns the path written to.
    """
    df = pd.DataFrame([row])
    header = not os.path.exists(path)

    try:
        df.to_csv(path, mode="a", index=False, header=header)
        return path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.replace(".csv", f"_{ts}.csv")
        df.to_csv(alt, mode="w", index=False, header=True)
        print(f"[Warning] '{path}' is locked (likely open in Excel). Wrote output to '{alt}' instead.\n")
        return alt


#Parsing (conversational,deterministic)

def parse_where_clause(text: str)->Dict[str,str]:
    filters = {}
    m = re.search(r"\bwhere\b(.+)$",text,flags=re.IGNORECASE)
    if not m:
        return filters
    clause = m.group(1).strip()
    parts = re.split(r"\band\b|,",clause,flags=re.IGNORECASE)
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            filters[k.strip()] = v.strip()
    return filters

def parse_days(text: str, default: int = 14)->int:
    m = re.search(r"last\s+(\d+)\s+day", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s+day", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return default

def parse_single_date(text: str)->Optional[str]:
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    return m.group(1) if m else None

def parse_date_range(text: str)->Tuple[Optional[str], Optional[str]]:
    """
    Supports:
      -from YYYY-MM-DD to YYYY-MM-DD
      -between YYYY-MM-DD and YYYY-MM-DD
      -YYYY-MM-DD to YYYY-MM-DD
    """
    m = re.search(r"\bfrom\s+(20\d{2}-\d{2}-\d{2})\s+to\s+(20\d{2}-\d{2}-\d{2})\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"\bbetween\s+(20\d{2}-\d{2}-\d{2})\s+and\s+(20\d{2}-\d{2}-\d{2})\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\s+to\s+(20\d{2}-\d{2}-\d{2})\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    return None, None

def infer_simple_filters(text: str, df: pd.DataFrame)->Dict[str, str]:
    """
    Lightweight inference for queries like:
    'for Home & Kitchen products'->{'Category': 'Home & Kitchen'}
    Priority: Category>Sub_category>Product_Name
    """
    text_l = text.lower()

    m = re.search(r"\bfor\s+(.+?)\s+products?\b", text_l)
    phrase = m.group(1).strip() if m else ""

    candidates = []
    if phrase:
        candidates.append(phrase)
    candidates.append(text_l)

    for col in ["Category", "Sub_category", "Product_Name"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).unique().tolist()
            vals = sorted(vals, key=lambda x: len(str(x)), reverse=True)
            for v in vals:
                v_l = str(v).lower()
                for hay in candidates:
                    if v_l and v_l in hay:
                        return {col: str(v)}
    return {}


#Core Data Engine

@dataclass
class DeviationResult:
    date: pd.Timestamp
    kpi: str
    actual: float
    baseline: float
    deviation_pct: float
    direction: str
    threshold: float
    scope_desc: str
    filters: Dict[str, str]

class KPIEngine:
    def __init__(self, df: pd.DataFrame, date_col: str, default_kpi: str, dims: List[str]):
        self.df = df.copy()
        self.date_col = date_col
        self.dim_cols = dims

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
        self.df = self.df.dropna(subset=[self.date_col]).sort_values(self.date_col)

        for c in self.df.columns:
            if c == self.date_col:
                continue
            if pd.api.types.is_numeric_dtype(self.df[c]):
                self.df[c] = safe_to_numeric(self.df[c])

        self.numeric_cols = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c]) and c!=self.date_col]
        self.kpi = default_kpi

    def set_kpi(self, kpi_col: str)->bool:
        if kpi_col in self.numeric_cols:
            self.kpi = kpi_col
            return True
        for c in self.numeric_cols:
            if normalize_col(c) == normalize_col(kpi_col):
                self.kpi = c
                return True
        return False

    def apply_filters(self, filters: Dict[str, str])->pd.DataFrame:
        d = self.df
        for k,v in (filters or {}).items():
            col = None
            for c in d.columns:
                if normalize_col(c) == normalize_col(k):
                    col = c
                    break
            if col is None:
                continue
            if not pd.api.types.is_numeric_dtype(d[col]):
                d = d[d[col].astype(str).str.lower().str.contains(str(v).lower(), na=False)]
            else:
                try:
                    d = d[d[col] == float(v)]
                except Exception:
                    pass
        return d

    def apply_date_range(self, d: pd.DataFrame, start: Optional[str], end: Optional[str])->pd.DataFrame:
        if start:
            d = d[d[self.date_col]>=pd.to_datetime(start)]
        if end:
            d = d[d[self.date_col] <= pd.to_datetime(end)]
        return d

    def daily_series(self, df: pd.DataFrame, col: str)->pd.DataFrame:
        return df.groupby(self.date_col, as_index=False)[col].sum()

    def trend(self, days: int, filters: Dict[str, str], start: Optional[str], end: Optional[str])->pd.DataFrame:
        d = self.apply_filters(filters)
        d = self.apply_date_range(d, start, end)
        if d.empty:
            return d
        if start or end:
            return self.daily_series(d, self.kpi)
        max_date = d[self.date_col].max()
        start_dt = max_date-pd.Timedelta(days=days-1)
        d = d[d[self.date_col]>=start_dt]
        return self.daily_series(d, self.kpi)

    def current_vs_baseline(self, filters: Dict[str, str], rolling_window: int, start: Optional[str], end: Optional[str])->Tuple[Optional[pd.Timestamp], Optional[float], Optional[float]]:
        d = self.apply_filters(filters)
        d = self.apply_date_range(d, start, end)
        if d.empty:
            return None, None, None
        daily = self.daily_series(d, self.kpi).sort_values(self.date_col).reset_index(drop=True)
        if daily.empty:
            return None,None,None
        values = daily[self.kpi].astype(float)
        base = values.rolling(window=rolling_window, min_periods=max(2, rolling_window // 2)).mean().shift(1)
        last_idx = len(daily)-1
        if pd.isna(base.iloc[last_idx]) or base.iloc[last_idx] == 0:
            return daily.loc[last_idx, self.date_col],float(values.iloc[last_idx]), None
        return daily.loc[last_idx, self.date_col],float(values.iloc[last_idx]), float(base.iloc[last_idx])

    def detect_deviations(self, days_back: int, rolling_window: int, threshold: float, filters: Dict[str, str], start: Optional[str], end: Optional[str])->List[DeviationResult]:
        d = self.apply_filters(filters)
        d = self.apply_date_range(d,start,end)
        if d.empty:
            return []

        daily = self.daily_series(d,self.kpi).sort_values(self.date_col).reset_index(drop=True)
        if daily.empty:
            return []

        if not (start or end):
            max_date = daily[self.date_col].max()
            start_dt = max_date-pd.Timedelta(days=days_back-1)
            daily = daily[daily[self.date_col]>=start_dt].reset_index(drop=True)

        values = daily[self.kpi].astype(float)
        base = values.rolling(window=rolling_window, min_periods=max(2, rolling_window // 2)).mean().shift(1)

        out = []
        for i in range(len(daily)):
            if pd.isna(base.iloc[i]) or pd.isna(values.iloc[i]):
                continue
            baseline = float(base.iloc[i])
            actual = float(values.iloc[i])
            if baseline == 0:
                continue
            dev = (actual-baseline) / baseline
            if abs(dev)>=threshold:
                direction = "drop" if dev<0 else "spike"
                scope_desc = "All data" if not filters else ", ".join([f"{k}={v}" for k, v in filters.items()])
                out.append(DeviationResult(
                    date=daily.loc[i, self.date_col],
                    kpi=self.kpi,
                    actual=actual,
                    baseline=baseline,
                    deviation_pct=float(dev),
                    direction=direction,
                    threshold=threshold,
                    scope_desc=scope_desc,
                    filters=filters
                ))
        return out

    #Causal analysis
    def _agg_driver(self,df: pd.DataFrame,col: str)->float:
        """
        Aggregation for daily driver:
        -Sum for volume/spend/count signals
        -Mean for score/ratio signals
        """
        n = normalize_col(col)
        s = safe_to_numeric(df[col])
        if any(x in n for x in ["sales", "spend", "no_rating", "count", "units"]):
            return float(s.sum(skipna=True))
        return float(s.mean(skipna=True))

    def causal_analysis(self, target_date: str, rolling_window: int, filters: Dict[str, str], top_n: int = 8)->pd.DataFrame:
        """
        Produces ranked hypotheses:
        -Numeric drivers: day vs baseline window (previous N days) using z-score if possible,else pct change.
        -Dimension contributions: biggest movers in revenue contribution by Category/Sub_category/Product.
        Ensures key drivers (Sales_m,Discount,ating) are included if present.
        """
        d = self.apply_filters(filters)
        if d.empty:
            return pd.DataFrame()

        td = pd.to_datetime(target_date)
        base_start = td-pd.Timedelta(days=rolling_window)
        base_end = td-pd.Timedelta(days=1)

        day_df = d[d[self.date_col] == td].copy()
        base_df = d[(d[self.date_col]>=base_start) & (d[self.date_col] <= base_end)].copy()
        if day_df.empty or base_df.empty:
            return pd.DataFrame()

        #Numeric drivers (daily aggregated series over baseline window)
        candidates = [c for c in self.numeric_cols if c!=self.kpi]
        rows = []

        base_days = sorted(base_df[self.date_col].unique())
        base_day_count = len(base_days) if base_days else 0

        for c in candidates:
            day_val = self._agg_driver(day_df, c)
            per_day = []
            for dt in base_days:
                chunk = base_df[base_df[self.date_col] == dt]
                per_day.append(self._agg_driver(chunk, c))
            per_day = np.array(per_day,dtype=float) if per_day else np.array([], dtype=float)

            if per_day.size == 0 or np.all(np.isnan(per_day)):
                continue

            base_mean = float(np.nanmean(per_day))
            base_std = float(np.nanstd(per_day, ddof=1)) if per_day.size>=2 else 0.0
            delta = float(day_val-base_mean)

            z = None
            pct = None
            if base_std>0:
                z = float((day_val-base_mean) / base_std)
            if base_mean!=0:
                pct = float((day_val-base_mean) / base_mean)

            corr = None
            try:
                kpi_series = []
                for dt in base_days:
                    chunk = base_df[base_df[self.date_col] == dt]
                    kpi_series.append(float(chunk[self.kpi].sum()))
                if len(kpi_series) == len(per_day) and len(kpi_series)>=5:
                    corr = float(np.corrcoef(np.array(per_day, dtype=float), np.array(kpi_series, dtype=float))[0, 1])
            except Exception:
                corr = None

            direction = "increased" if delta>0 else "decreased"
            if corr is None:
                corr_txt = "unknown relationship"
            else:
                corr_txt = "positive relationship" if corr>0.2 else ("negative relationship" if corr<-0.2 else "weak relationship")

            expl = f"{c} {direction} vs baseline ({fmt_num(base_mean)} → {fmt_num(day_val)})."
            if pct is not None:
                expl += f" Change: {fmt_pct(pct)}."
            if corr is not None:
                expl += f" Baseline correlation with KPI: {corr:+.2f} ({corr_txt})."

            # rank score
            rank = abs(z) if z is not None else (abs(pct) * 10 if pct is not None else abs(delta))
            rows.append({
                "driver_type": "numeric",
                "driver": c,
                "day_value": float(day_val),
                "baseline_value": float(base_mean),
                "delta": float(delta),
                "pct_change": None if pct is None else float(pct),
                "z_score": None if z is None else float(z),
                "baseline_corr_with_kpi": corr,
                "rank_score": float(rank),
                "explain": expl
            })

        key_order = ["Sales_m", "Discount", "Rating"]
        for key in key_order:
            if key in self.df.columns and key in candidates:
                if not any(r["driver"] == key for r in rows):
                    day_val = self._agg_driver(day_df, key)
                    base_days = sorted(base_df[self.date_col].unique())
                    per_day = [self._agg_driver(base_df[base_df[self.date_col] == dt], key) for dt in base_days]
                    per_day = np.array(per_day, dtype=float) if per_day else np.array([], dtype=float)
                    if per_day.size:
                        base_mean = float(np.nanmean(per_day))
                        base_std = float(np.nanstd(per_day, ddof=1)) if per_day.size>=2 else 0.0
                        delta = float(day_val-base_mean)
                        z = float((day_val-base_mean) / base_std) if base_std>0 else None
                        pct = float((day_val-base_mean) / base_mean) if base_mean!=0 else None
                        rank = abs(z) if z is not None else (abs(pct) * 10 if pct is not None else abs(delta))
                        rows.append({
                            "driver_type": "numeric",
                            "driver": key,
                            "day_value": float(day_val),
                            "baseline_value": float(base_mean),
                            "delta": float(delta),
                            "pct_change": None if pct is None else float(pct),
                            "z_score": None if z is None else float(z),
                            "baseline_corr_with_kpi": None,
                            "rank_score": float(rank),
                            "explain": f"{key} moved vs baseline ({fmt_num(base_mean)} → {fmt_num(day_val)})."
                        })

        dim_rows = []
        for dim in ["Category", "Sub_category", "Product_Name"]:
            if dim not in self.dim_cols or dim not in day_df.columns:
                continue
            day_g = day_df.groupby(dim)[self.kpi].sum()
            base_g = base_df.groupby(dim)[self.kpi].sum() / max(1, base_df[self.date_col].nunique())
            merged = pd.concat([day_g.rename("day"), base_g.rename("baseline")], axis=1).fillna(0.0)
            merged["delta"] = merged["day"]-merged["baseline"]

            top_movers = merged.reindex(merged["delta"].abs().sort_values(ascending=False).head(3).index)
            for idx, r in top_movers.iterrows():
                dim_rows.append({
                    "driver_type": f"dimension:{dim}",
                    "driver": f"{dim}={idx}",
                    "day_value": float(r["day"]),
                    "baseline_value": float(r["baseline"]),
                    "delta": float(r["delta"]),
                    "pct_change": None,
                    "z_score": None,
                    "baseline_corr_with_kpi": None,
                    "rank_score": float(abs(r["delta"])),
                    "explain": f"Contribution shift for {dim}='{idx}' vs baseline."
                })

        out = pd.DataFrame(rows + dim_rows)
        if out.empty:
            return out
        out = out.sort_values("rank_score",ascending=False).head(top_n).reset_index(drop=True)
        return out


#Action Agent

CAUSE_ACTIONS = {
    "sales_m": {
        "title": "Sales volume changed",
        "actions": [
            "Increase demand: boost targeted campaigns for the affected scope and refresh creatives.",
            "Improve conversion: audit PDP/checkout funnel; run quick A/B tests on top SKUs.",
            "Check inventory/availability to rule out stockouts for high-selling products."
        ]
    },
    "discount": {
        "title": "Discount strategy shifted",
        "actions": [
            "If discount decreased and volume fell: run short tactical promos/bundles to regain momentum.",
            "If discount increased: reduce blanket discounting; switch to targeted offers to protect margin.",
            "Review price elasticity for top categories/products and adjust discount depth accordingly."
        ]
    },
    "rating": {
        "title": "Customer rating/feedback shifted",
        "actions": [
            "Investigate product issues (defects/returns) and fix top complaint themes quickly.",
            "Improve CX: shipping/packaging/returns for affected products and lanes.",
            "Merchandise top-rated alternatives and highlight review snippets in listings."
        ]
    },
    "m_spend": {
        "title": "Marketing spend changed",
        "actions": [
            "Reallocate spend toward categories with volume loss; pause low-return campaigns.",
            "Increase bids/placements temporarily for the affected scope and monitor CAC/ROAS daily.",
            "Refresh creatives and landing pages; run 2–3 fast experiments."
        ]
    },
    "supply_chain_e": {
        "title": "Supply chain efficiency shifted",
        "actions": [
            "Check fulfillment SLA,delivery delays,and stockouts; prioritize replenishment for top SKUs.",
            "Coordinate with logistics for affected lanes and enforce escalation rules.",
            "Add daily monitoring for OOS rate and delivery performance in the affected scope."
        ]
    },
    "market_t": {
        "title": "Market trend shifted",
        "actions": [
            "Align merchandising and ads to trend: push trending sub-categories and pause weak demand SKUs.",
            "Review competitor pricing/promos and adjust offers where needed.",
            "Update assortment and search/collection placements based on trend movement."
        ]
    },
    "seasonality_t": {
        "title": "Seasonality factor moved",
        "actions": [
            "Adjust forecasts and promo calendar to seasonal demand; plan bundles around seasonal needs.",
            "Ensure inventory readiness for seasonal movers; reduce exposure to off-season items.",
            "Update onsite banners/search ranking to match seasonal intent."
        ]
    }
}


class ActionAgent:
    def map_driver_to_actions(self, driver: str)->Tuple[str, List[str]]:
        key = normalize_col(driver)
        for k, v in CAUSE_ACTIONS.items():
            if k in key:
                return v["title"], v["actions"]

        #Dimension drivers
        if key.startswith("category="):
            val = driver.split("=", 1)[1]
            return "Category mix shift", [
                f"Prioritize merchandising and marketing for '{val}' for the next 7 days (homepage/search placements, collections).",
                f"Review pricing/discounts and promo calendar for '{val}' to restore volume.",
                f"Check inventory depth and fulfillment performance for top '{val}' SKUs."
            ]
        if key.startswith("sub_category="):
            val = driver.split("=", 1)[1]
            return "Sub-category mix shift", [
                f"Boost visibility and targeted promotions for sub-category '{val}'.",
                f"Audit top SKUs in '{val}' for stockouts, pricing, and reviews.",
                f"Run a short campaign test and monitor uplift daily."
            ]
        if key.startswith("product_name="):
            val = driver.split("=", 1)[1]
            return "Product-level shift", [
                f"Check availability and delivery timelines for '{val}' (stockouts or delays can cause sudden drops).",
                f"Review listing quality for '{val}' (images, title, price, discount, reviews) and optimize quickly.",
                f"Run product-specific ads or bundles if the product is strategic."
            ]

        return "General performance shift", [
            "Validate the top drivers above and run targeted tests (promo, spend, merchandising) in the affected scope.",
            "Monitor daily for 7 days and iterate based on results."
        ]

    def build_email(self, deviation: DeviationResult, drivers: pd.DataFrame)->Tuple[str, str]:
        subj_kpi = deviation.kpi.replace("_", " ")
        subject = f"{subj_kpi} {deviation.direction.title()} ({fmt_pct(deviation.deviation_pct)}) on {deviation.date.date()} – Recommended Actions"

        lines = []
        lines.append(f"Scope: {deviation.scope_desc}")
        lines.append(f"KPI: {deviation.kpi}")
        lines.append(f"Actual: {fmt_num(deviation.actual)}")
        lines.append(f"Baseline (prev window): {fmt_num(deviation.baseline)}")
        lines.append(f"Deviation: {fmt_pct(deviation.deviation_pct)} ({deviation.direction})")
        lines.append("")
        lines.append("Top suspected drivers (ranked):")
        for i, r in drivers.iterrows():
            extra = ""
            if pd.notna(r.get("pct_change", np.nan)):
                extra = f" | change {fmt_pct(float(r['pct_change']))}"
            lines.append(f"{i+1}. {r['driver']} (Δ={fmt_num(r['delta'])}{extra})")
        lines.append("")
        lines.append("Prioritized recommended actions:")

        used_blocks = 0
        for _, r in drivers.head(3).iterrows():
            title, acts = self.map_driver_to_actions(str(r["driver"]))
            lines.append(f"- {title}:")
            for a in acts[:3]:
                lines.append(f"  • {a}")
            used_blocks += 1
            lines.append("")

        lines.append("Next steps:")
        lines.append("• Validate impact within the scoped slice and monitor daily for the next 7 days.")
        lines.append("• If the deviation persists for 3+ days, escalate with a deeper investigation by channel/SKU/region (if available).")

        return subject, "\n".join(lines)


#CLI Application

HELP = """
Commands (examples):
- help
- list columns
- list kpis
- set kpi Overall_Revenue
- show trend last 14 days
- show trend for Home & Kitchen products over last 14 days
- show trend last 14 days where Category=Home & Kitchen
- show trend last 14 days from 2025-12-01 to 2025-12-14
- summary last 7 days
- compare last 7 days vs previous 7 days
- run monitoring last 30 days
- set threshold 0.15
- set window 7
- explain last alert
- why did revenue drop on 2026-01-15?
- why did Sales_m drop on 2026-01-15?
- exit
""".strip()


class CLIApp:
    def __init__(self, engine: KPIEngine, alerts_path: str, recs_path: str):
        self.engine = engine
        self.alerts_path = alerts_path
        self.recs_path = recs_path
        self.threshold = 0.15
        self.rolling_window = 7
        self.last_alert: Optional[DeviationResult] = None
        self.action_agent = ActionAgent()

    def _resolve_filters(self, text: str)->Dict[str, str]:
        filters = parse_where_clause(text)
        if not filters:
            filters.update(infer_simple_filters(text, self.engine.df))
        return filters

    def _resolve_date_range(self, text: str)->Tuple[Optional[str], Optional[str]]:
        return parse_date_range(text)

    def _scope_desc(self, filters: Dict[str, str])->str:
        return "All data" if not filters else ", ".join([f"{k}={v}" for k, v in filters.items()])

    def start(self):
        print("\nAgentic KPI Intelligence (CLI)")
        print("--------------------------------")
        print(f"Date column: {self.engine.date_col}")
        print(f"Active KPI: {self.engine.kpi}")
        print(f"Dimensions: {', '.join(self.engine.dim_cols) if self.engine.dim_cols else '(none)'}")
        print("\nType 'help' for commands.\n")

        while True:
            try:
                text = input("kpi-agent> ").strip()
            except (EOFError,KeyboardInterrupt):
                print("\nExiting.")
                return

            if not text:
                continue
            t = text.lower()

            if t in ("exit", "quit"):
                print("Bye.")
                return
            if t == "help":
                print(HELP + "\n")
                continue
            if t == "list columns":
                print("\nColumns:")
                for c in self.engine.df.columns:
                    print(f"- {c}")
                print("")
                continue
            if t == "list kpis":
                print("\nAvailable KPI columns (numeric):")
                for c in self.engine.numeric_cols:
                    print(f"- {c}")
                print("")
                continue
            if t.startswith("set kpi"):
                m = re.search(r"set kpi\s+(.+)$",text,flags=re.IGNORECASE)
                if not m:
                    print("Usage: set kpi <column>\n")
                    continue
                kpi = m.group(1).strip()
                if self.engine.set_kpi(kpi):
                    print(f"Active KPI set to: {self.engine.kpi}\n")
                else:
                    print("KPI not found. Use 'list kpis' to see available numeric columns.\n")
                continue
            if t.startswith("set threshold"):
                m = re.search(r"set threshold\s+([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
                if m:
                    self.threshold = float(m.group(1))
                    print(f"Threshold set to {self.threshold:.2f}\n")
                else:
                    print("Usage: set threshold 0.15\n")
                continue
            if t.startswith("set window"):
                m = re.search(r"set window\s+(\d+)", text, flags=re.IGNORECASE)
                if m:
                    self.rolling_window = int(m.group(1))
                    print(f"Rolling window set to {self.rolling_window} days\n")
                else:
                    print("Usage: set window 7\n")
                continue

            #Intent routing (chatbot-style)
            if t.startswith("show") and "trend" in t:
                self.cmd_trend(text)
            elif t.startswith("summary"):
                self.cmd_summary(text)
            elif t.startswith("compare"):
                self.cmd_compare(text)
            elif t.startswith("run monitoring"):
                self.cmd_monitor(text)
            elif t.startswith("explain last alert"):
                self.cmd_explain_last_alert()
            elif t.startswith("why"):
                self.cmd_why(text)
            else:
                print("Not sure how to answer. Type 'help'.\n")

    def cmd_trend(self,text: str):
        days = parse_days(text,14)
        start,end = self._resolve_date_range(text)
        filters = self._resolve_filters(text)

        tr = self.engine.trend(days,filters,start,end)
        if tr.empty:
            print("No data found for given filters/date range.\n")
            return

        scope = self._scope_desc(filters)
        kpi = self.engine.kpi
        print(f"\n{kpi} trend ({'date range' if (start or end) else f'last {days} days'}) | Scope: {scope}")
        tr = tr.sort_values(self.engine.date_col)
        for _, r in tr.iterrows():
            print(f"- {pd.to_datetime(r[self.engine.date_col]).date()}: {fmt_num(float(r[kpi]))}")
        print("")

        latest_date = tr[self.engine.date_col].max()
        latest_row = tr[tr[self.engine.date_col] == latest_date].iloc[0]
        actual = float(latest_row[kpi])

        devs = self.engine.detect_deviations(days_back=days, rolling_window=self.rolling_window, threshold=self.threshold, filters=filters, start=start, end=end)
        if devs:
            latest = devs[-1]
            print(f"ALERT: {kpi} {'dropped' if latest.deviation_pct<0 else 'increased'} by {abs(latest.deviation_pct)*100:.1f}% compared to {self.rolling_window}-day rolling average ({fmt_num(latest.baseline)}). Threshold={self.threshold:.2f}\n")
            self.trigger_alert(latest)
        else:
            _, _, baseline = self.engine.current_vs_baseline(filters, self.rolling_window, start, end)
            if baseline is not None:
                dev = (actual-baseline) / baseline if baseline!=0 else np.nan
                print(f"Latest vs baseline: actual={fmt_num(actual)}, baseline={fmt_num(baseline)}, change={fmt_pct(dev)}\n")
            else:
                print("No deviations detected (or insufficient prior days for baseline).\n")

    def cmd_summary(self, text: str):
        days = parse_days(text, 7)
        start, end = self._resolve_date_range(text)
        filters = self._resolve_filters(text)

        tr = self.engine.trend(days, filters, start, end)
        if tr.empty:
            print("No data found.\n")
            return
        kpi = self.engine.kpi
        total = float(tr[kpi].sum())
        avg = float(tr[kpi].mean())
        latest = float(tr.sort_values(self.engine.date_col).iloc[-1][kpi])

        print(f"\nSummary ({'date range' if (start or end) else f'last {days} days'}) | KPI: {kpi} | Scope: {self._scope_desc(filters)}")
        print(f"- Total: {fmt_num(total)}")
        print(f"- Avg per day: {fmt_num(avg)}")
        print(f"- Latest day: {fmt_num(latest)}\n")

    def cmd_compare(self,text: str):
        days = parse_days(text,7)
        start, end = self._resolve_date_range(text)
        filters = self._resolve_filters(text)

        d = self.engine.apply_filters(filters)
        d = self.engine.apply_date_range(d,start,end)
        if d.empty:
            print("No data found.\n")
            return

        daily = self.engine.daily_series(d,self.engine.kpi).sort_values(self.engine.date_col).reset_index(drop=True)
        if len(daily)<days * 2:
            print("Not enough history to compare last period vs previous period.\n")
            return

        kpi = self.engine.kpi
        last = daily.iloc[-days:][kpi].sum()
        prev = daily.iloc[-2*days:-days][kpi].sum()
        if prev == 0:
            print("Previous period is 0; cannot compute % change.\n")
            return
        pct = (last-prev) / prev

        print(f"\nCompare | KPI: {kpi} | Scope: {self._scope_desc(filters)}")
        print(f"- Last {days} days: {fmt_num(float(last))}")
        print(f"- Previous {days} days: {fmt_num(float(prev))}")
        print(f"- Change: {fmt_pct(float(pct))}\n")

    def cmd_monitor(self,text: str):
        days = parse_days(text,30)
        start,end = self._resolve_date_range(text)
        filters = self._resolve_filters(text)

        devs = self.engine.detect_deviations(days_back=days,rolling_window=self.rolling_window,threshold=self.threshold,filters=filters,start=start,end=end)
        if not devs:
            print(f"No deviations in {'date range' if (start or end) else f'last {days} days'} (threshold={self.threshold:.2f}).\n")
            return

        print(f"\nFound {len(devs)} deviation(s) in {'date range' if (start or end) else f'last {days} days'} | KPI: {self.engine.kpi} | Scope: {self._scope_desc(filters)}")
        for d in devs:
            print(f"- {d.date.date()}: {d.direction} {fmt_pct(d.deviation_pct)} (actual={fmt_num(d.actual)}, baseline={fmt_num(d.baseline)})")
        print("")

        self.trigger_alert(devs[-1])

    def cmd_explain_last_alert(self):
        if not self.last_alert:
            print("No alert yet. Run monitoring first.\n")
            return
        self.explain_and_recommend(self.last_alert)

    def cmd_why(self, text: str):
        dt = parse_single_date(text)
        if not dt:
            print("Include a date like YYYY-MM-DD.\n")
            return

        m = re.search(r"why did\s+(.+?)\s+(drop|spike|increase|decrease)\s+on", text, flags=re.IGNORECASE)
        prev_kpi = self.engine.kpi
        if m:
            candidate = m.group(1).strip()
            self.engine.set_kpi(candidate)

        start, end = self._resolve_date_range(text)
        filters = self._resolve_filters(text)

        d = self.engine.apply_filters(filters)
        d = self.engine.apply_date_range(d,start,end)
        daily = self.engine.daily_series(d,self.engine.kpi).sort_values(self.engine.date_col).reset_index(drop=True)
        td = pd.to_datetime(dt)
        row = daily[daily[self.engine.date_col] == td]
        if row.empty:
            print("No data for that date/scope.\n")
            self.engine.set_kpi(prev_kpi)
            return

        idx = row.index[0]
        values = daily[self.engine.kpi].astype(float)
        base = values.rolling(window=self.rolling_window, min_periods=max(2, self.rolling_window // 2)).mean().shift(1)

        baseline = float(base.iloc[idx]) if not pd.isna(base.iloc[idx]) else np.nan
        actual = float(row.iloc[0][self.engine.kpi])
        if pd.isna(baseline) or baseline == 0:
            print("Not enough prior days to compute baseline.\n")
            self.engine.set_kpi(prev_kpi)
            return

        dev = (actual-baseline) / baseline
        direction = "drop" if dev<0 else "spike"
        pseudo = DeviationResult(td,self.engine.kpi,actual,baseline,float(dev),direction,self.threshold,self._scope_desc(filters),filters)
        self.explain_and_recommend(pseudo)

        self.engine.set_kpi(prev_kpi)

    def trigger_alert(self, dev: DeviationResult):
        self.last_alert = dev
        append_row_csv(self.alerts_path, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "date": str(dev.date.date()),
            "kpi": dev.kpi,
            "actual": dev.actual,
            "baseline": dev.baseline,
            "deviation_pct": dev.deviation_pct,
            "direction": dev.direction,
            "threshold": dev.threshold,
            "scope": dev.scope_desc,
            "filters": str(dev.filters),
        })
        print("[Logged] alerts_log.csv updated.\n")
        self.explain_and_recommend(dev)

    def explain_and_recommend(self, dev: DeviationResult):
        print(f"Deviation Explanation ({dev.date.date()}) | KPI: {dev.kpi} | Scope: {dev.scope_desc}")
        print(f"- Actual:   {fmt_num(dev.actual)}")
        print(f"- Baseline: {fmt_num(dev.baseline)} (prev {self.rolling_window}-day rolling avg)")
        print(f"- Deviation: {fmt_pct(dev.deviation_pct)} ({dev.direction})\n")

        drivers = self.engine.causal_analysis(str(dev.date.date()), self.rolling_window, dev.filters, top_n=8)
        if drivers.empty:
            print("Causal analysis: insufficient signal.\n")
            return

        print("Top suspected drivers (ranked):")
        for i, r in drivers.iterrows():
            z = r.get("z_score",None)
            pct = r.get("pct_change",None)
            ztxt = f"z={z:.2f}" if z is not None and pd.notna(z) else ""
            ptxt = f"chg={fmt_pct(float(pct))}" if pct is not None and pd.notna(pct) else ""
            meta = " | ".join([x for x in [ztxt, ptxt] if x])
            if meta:
                meta = " | " + meta
            print(f"{i+1}. {r['driver']} | Δ={fmt_num(r['delta'])}{meta}")
            print(f"-{r['explain']}")
        print("")

        subject,body = self.action_agent.build_email(dev,drivers)
        print("Email-ready recommendations:")
        print(f"Subject: {subject}\n")
        print(body)
        print("")

        append_row_csv(self.recs_path,{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "date": str(dev.date.date()),
            "kpi": dev.kpi,
            "deviation_pct": dev.deviation_pct,
            "scope": dev.scope_desc,
            "subject": subject,
            "body": body.replace("\n", " | "),
            "top_drivers": "; ".join([str(x) for x in drivers["driver"].tolist()]),
        })
        print("[Logged] recommendations.csv updated.\n")


#Entry

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="kpi_data.csv", help="Input dataset CSV (provided during development phase)")
    p.add_argument("--alerts", default="alerts_log.csv")
    p.add_argument("--recs", default="recommendations.csv")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Failed to read dataset: {e}")
        sys.exit(1)

    df, date_col, date_note = detect_date_col_or_create(df, force_days=90)
    df, rev_col, rev_note = detect_revenue_col_or_create(df)

    #dimensions and default KPI
    dims = get_dimension_cols(df, date_col)
    engine = KPIEngine(df, date_col=date_col, default_kpi=rev_col, dims=dims)

    print("\n[Dataset setup]")
    print(date_note)
    print(rev_note)
    print("")

    app = CLIApp(engine,args.alerts,args.recs)
    app.start()


if __name__ == "__main__":
    main()
