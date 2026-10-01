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
| `data/weatherAUS.csv` | Snapshot of the rattle weatherAUS file (Bureau of Meteorology data, to 30 Jan 2026) |
| `data/bom_*.csv` | BOM Daily Weather Observations, Jan–Sep 2026, five cities |
| `figures/`, `results/` | Figures and numbers used in the journal |
| `results/history_seq_cache.json` | Saved results of the 30- and 90-day GRU/Transformer runs (loaded on Colab so they are not trained again) |
| `requirements.txt` | Library versions used to produce the results |

## Data sources

- Williams, G. weatherAUS, rattle project: https://rattle.togaware.com/weatherAUS.csv
- Bureau of Meteorology, Daily Weather Observations: http://www.bom.gov.au/climate/dwo/
  (© Commonwealth of Australia, Bureau of Meteorology)

## Image credits

- `figures/web/boosting.png`: Sirakorn, "Ensemble Boosting", Wikimedia Commons, CC BY-SA 4.0.
- `figures/web/transformer.png`, `figures/web/gru_cell.png`: Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (d2l.ai), CC BY-SA 4.0.
- All other figures were made for this project.
