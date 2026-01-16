# KPI Intelligence & Action Agent
## An intelligent internal tool for business performance monitoring


It supports:

* Track Overall Revenue and key KPIs daily 
* Support filtering by dimensions (Product, Category, Date Range, etc.)
* Deviation detection vs rolling baseline
* Causal hypothesis ranking (numeric + dimension contribution shifts)
* Present recommendations in an email-ready format
* CLI+ Streamlit dashboard


Files:

* main.py- CLI entry point
* kpi_data.csv- provided dataset 
* alerts_log.csv- alert log output
* recommendations.csv- recommendation output
* app.py- Streamlit UI


Setup:

1. Go to the file environment
2. Run the following
* pip install -r requirements.txt (For installing dependencies)
* python main.py --data kpi_data.csv (To see in CLI)
* streamlit run app.py (To see in streamlit)


Project Structure:

kpi_agentic (Main Folder)-

* alerts_log.csv
* app.py
* kpi_data.csv
* main.py
* README.md
* recommendations.csv
* requirements.txt


Some CLI Commands:

* show trend for Home & Kitchen products over last 14 days
* run monitoring last 30 days
* explain last alert
* set threshold 0.10
* set window 7
* show trend last 14 days from 2025-12-01 to 2025-12-14
* list kpis
* set kpi Sales_m
* show trend last 14 days where Category=Home & Kitchen
