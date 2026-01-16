# KPI Intelligence & Action Agent🤖
## An intelligent internal tool for business performance monitoring

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red.svg)


## Features✨:

* Track Overall Revenue and key KPIs daily 
* Support filtering by dimensions (Product, Category, Date Range, etc.)
* Deviation detection vs rolling baseline
* Causal hypothesis ranking (numeric + dimension contribution shifts)
* Present recommendations in an email-ready format
* CLI+ Streamlit dashboard


## Quick Start🚀:

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/devyanighildiyal/kpi-agentic-intelligence.git
   cd kpi-agentic-intelligence
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run main_app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`


## Project Structure📁:

```
kpi_agentic/
├── alerts_log.csv               
├── app.py            
├── kpi_data.csv              
├── main.py           
├── README.md        
├── recommendations.csv             
├── requirements.txt        
            
```

## Example CLI Commands 🔧:

* show trend for Home & Kitchen products over last 14 days
* run monitoring last 30 days
* explain last alert
* set threshold 0.10
* set window 7
* show trend last 14 days from 2025-12-01 to 2025-12-14
* list kpis
* set kpi Sales_m
* show trend last 14 days where Category=Home & Kitchen
