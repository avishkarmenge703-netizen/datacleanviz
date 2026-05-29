# DataCleanViz

Automated data cleaning and comprehensive EDA for structured datasets (CSV/Excel). Provides a Streamlit web app for easy interaction.

## Features
- Smart encoding detection
- Configurable missing value imputation (median/mean/mode/Unknown)
- Outlier detection and capping (IQR / Z-score)
- Automatic date parsing, whitespace trimming, case fixing
- Drops columns with >50% missing values
- Interactive EDA dashboard (histograms, KDE, count plots, box plots, heatmap, clustermap, pair plots, stacked bar charts)
- Detailed cleaning log and downloadable cleaned CSV

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/datacleanviz.git
cd datacleanviz
pip install -r requirements.txt
