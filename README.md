<div align="center">
  <img src="assets/images/wordart (1).png" style="max-width: 100%;" alt="NFL Draft Comparison Cards" />
</div>

NFL Draft prospect comparison cards built with a k-nearest neighbor (KNN) algorithm. Takes raw combine measurements and college production stats, calculates per-season percentiles, then uses Manhattan distance to find each prospect's 3 most similar historical players.

<div align="center">
  <img src="https://raw.githubusercontent.com/array-carpenter/nfl_draft_app/master/output/draft_cards/Sonny_Styles_pre_combine.png" width="80%" />
</div>

## How It Works

1. Combine and pro day measurements are merged with college production stats
2. Volume stats (yards, TDs, tackles) are normalized per season played
3. Percentile rankings are computed within each position group
4. KNN with Manhattan distance finds the 3 most similar historical players
5. A comparison card is generated with radar charts and a stat table

## Run Locally

```bash
uv sync
uv run main.py
```

Edit `main.py` to change the target player.

## Streamlit App

```bash
uv run streamlit run app.py
```

Search for any prospect in the database to generate their comparison card on the fly.

## Data

Combine and pro day data sourced from the [nfl-draft-data](https://github.com/array-carpenter/nfl-draft-data) repo. College production stats from CFBD.
