# Batter wOBA Projection

An empirical Bayes shrinkage model to project wOBA for MLB hitters in the first half of 2026 using public Statcast and Baseball Reference data from 2024 and 2025. Please refer to the PDF in the repo for the full report.

## Overview

This project was originally an assignment to “create batter projections for the first half of the 2026 season from a dataset of recent batter plate appearances.” Specifically, I was given one week to build a model to predict each player’s wOBA for the first half of 2026 (2026 1H) using plate appearance data from 2024 and 2025. Additionally, accuracy was not the sole priority as I was required to provide analysis of the model results including how some players' predictions were driven.

Training data was provided via a CSV containing 2024 and 2025 plate appearances and a CSV with overall player statistics by season-half (ex: 2024 1H). This data was similar to real MLB data, but did not match perfectly. In order to publish this project, I’ve replaced the CSVs that provided the original data with real MLB data using the pybaseball package. 


## Data

### Sources
All data is downloaded via the pybaseball package. Statcast data is downloaded for pitch-level statistics while Baseball Reference data is downloaded for box-score counts. The Chadwick register is also used for player names.

Only regular season data from 2024 and 2025 are used. The first half of a season is defined as all games before or on June 30, while the second half is all games on or after July 1. All players with at least 1 PA in the first half of 2026 are given a prediction.

### The player-half panel
This is the dataset I use for model training, built from the sources above. Each row is one hitter in one half-season and it contains outcome counts, pitch-level counts, batted-ball counts, and the counts behind every candidate feature tested for this model.


## Model

The model is split into two stages: one to predict the outcome of a PA (K, BB, HBP, or BIP) and one to predict the outcome of a Ball In Play (1B, 2B, 3B, HR, or OUT). This split exists because all BIP outcomes are dependent on contact, and this structure isolates contact skill to the PA stage, allowing the BIP stage to focus on the conditional result. The two stages have otherwise identical designs.

For both stages, each outcome (K, BB, 1B, etc.) has its own shrinkage model with a “prior” based on informative traits (whiff%, exit velo, etc.). Since the shrinkage target is fitted from the same hitter's data that his outcome counts are pooled from, his record enters both the prior and the likelihood, so the prior is not independent of the likelihood and the pure Bayesian framework fails. In other words, a hitter's traits are strongly correlated with his own outcomes, so his history is effectively counted twice, which invalidates a Bayesian (posterior) interpretation of this model.

This also means kappa is a regularization strength rather than a prior sample size. However, the point predictions remain valid because every fitted parameter is forward-looking.

Please refer to the PDF report for a full breakdown of the model.


## Results

To test the model, it was fitted three times, once seeing data from only 2024, once seeing data from 2025 1H as well, and the final model seeing all available data. While the coefficients of the model change every fold, this keeps the system honest by completely excluding data from during or after scoring. I compared my model to 3 controls, all calculated using PA-weighted MSE on observed wOBA (times 1000): league average, own unshrunk wOBA, and direct shrinkage of wOBA - my realistic floor model for the project. The final results follow:

| Method | 2025 1H | 2025 2H | Spread | Mean | vs floor |
| :----- | ------: | ------: | -----: | ---: | -------: |
| League average | 2.56 | 2.53 | 0.02 | 2.55 | 23.6% | 
| Own unshrunk wOBA | 2.90 | 2.71 | 0.19 | 2.81 | 36.2% | 
| Direct shrinkage of wOBA | 2.11 | 2.02 | 0.09 | 2.06 | --- |
| Two-stage final model | 2.07 | 1.81 | 0.26 | 1.94 | -6.0% |

The final model easily beats the naive baselines of league average and previous player wOBA while also beating the direct wOBA shrinkage by 6%.

Here are the top 10 projected hitters by wOBA for the first half of 2026. 

| Rank | Player | Projected wOBA |
| ---: | :----- | -------------: |
| 1 | Aaron Judge | .451 |
| 2 | Shohei Ohtani | .415 |
| 3 | Yordan Alvarez | .411 |
| 4 | Juan Soto | .409 |
| 5 | Vladimir Guerrero Jr. | .390 |
| 6 | Bobby Witt Jr. | .386 |
| 7 | Ketel Marte | .382 |
| 8 | Kyle Schwarber | .378 |
| 9 | Kyle Tucker | .373 |
| 10 | Corey Seager | .371 |


## Repository structure

| File | Contents |
| :--- | :------- |
| `wOBA Projection with Empirical Bayes Shrinkage.pdf` | The full report: model design, results, and analysis. |
| `woba_projection_notebook.ipynb` | The main notebook: data exploration, configuration, fitting, validation, rejected proposals, and the 2026 1H projections. |
| `data.py` | Downloads the data and builds the player-half panel, the holdout list, and the player names table. Knows nothing about the model. |
| `model.py` | The estimator: pooled history, feature shrinkage, the logit prior, and the shrinkage step. No fitting. |
| `tuning.py` | All parameter fitting and the expanding window mechanism that keeps every fit on halves before its target. |
| `evaluate.py` | Composes the two stages into wOBA, scores predictions, and builds the validation and player tables. |
| `surface.py` | The xwOBAcon surface used only by the Stage XW comparison in Rejected Proposals. |

## Running it

### Requirements

This was built on Python 3.14 with the following packages: pandas, numpy, scipy, scikit-learn, matplotlib, pybaseball, pyarrow, and jupyter.

An internet connection is required, since all data is downloaded through pybaseball.

### Reproducing the projections

Open `woba_projection_notebook.ipynb` from the repository folder and run all cells.

On the first run, the Statcast pull for 2024–2025 takes about 5 minutes. It is saved to `cache/` as a Parquet file (about 220 MB), and later runs read that file instead of downloading again. To force a fresh download, call `data.load_statcast(..., refresh=True)`.

The projections are written to `predictions.csv` (one row per hitter: `BatterID`, `pred_wOBA`) by the cell in the "2026 1H projection" section.

## Future work

There are many features that I did not have the time and/or data to include do to the original constraints of this project. An aging curve and park factors represent, in my opinion, the most desirable fruit left on the tree. Public aging curves use decades of career data to shape projected improvement or decline by age. Not only do I only have 2 years of data (less if I want to test responsibly), but this project is not concerned with long term projections. That being said, with more time this would have been the next item I built and tested. Park factors, on the other hand, are entirely limited by the data I was originally provided. The CSVs I relied on when I built this model did not have a team variable, which made pursuing park factors a non-starter.

## Author

Please reach out to me, Simon Todreas, with any questions or comments on my [LinkedIn!](https://www.linkedin.com/in/simon-todreas/)
