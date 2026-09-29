# Will it rain tomorrow? A deployable rain forecast for Australian weather stations

UTS 32513/31005 Advanced Data Analytics Algorithms, Machine Learning – Assessment 2 (Option 2)
Ngoc Dang Nguyen (26059877)

At 3:30 pm on day *t*, the model predicts the probability that more than 1 mm of rain falls between 9 am on day *t+1*
and 9 am on day *t+2*, using only readings that are known at 3:30 pm.

## Run

Open `rain_forecast_au.ipynb` in Google Colab and choose **Runtime → Run all** (about 10–15 minutes on CPU).
The notebook downloads its data from the original sources and uses the copies in `data/` if a source is not reachable.

## Contents

| Path | What it is |
|---|---|
| `rain_forecast_au.ipynb` | Full implementation with outputs |
| `rain_forecast_au.py` | The same notebook as a plain Python script (jupytext percent format) |
| `data/weatherAUS.csv` | Snapshot of the rattle weatherAUS file (Bureau of Meteorology data, to 30 Jan 2026) |
| `data/bom_*.csv` | BOM Daily Weather Observations, Jan–Sep 2026, five cities |
| `figures/`, `results/` | Figures and numbers used in the journal |
| `requirements.txt` | Library versions used to produce the results |

## Data sources

- Williams, G. weatherAUS, rattle project: https://rattle.togaware.com/weatherAUS.csv
- Bureau of Meteorology, Daily Weather Observations: http://www.bom.gov.au/climate/dwo/
  (© Commonwealth of Australia, Bureau of Meteorology)
