# IPL Player Performance Predictor

A machine learning system to predict cricket player performance in the Indian Premier League (IPL) based on historical data from 2008-2024.

## 📌 Overview

This project implements a comprehensive machine learning pipeline to analyze and predict IPL player performance. The system:
- Processes ball-by-ball IPL match data
- Generates advanced cricket-specific performance metrics
- Trains separate Random Forest models for batsmen and bowlers
- Provides detailed performance predictions with confidence intervals

## ✨ Features

- **Comprehensive Data Processing**
  - Optimized memory handling with dtype configuration
  - Smart handling of missing values
  - Minimum participation filters (10 balls for batsmen, 30 balls for bowlers)

- **Advanced Cricket Metrics**
  - Consistency scores based on coefficient of variation
  - Weighted recent form metrics
  - Boundary percentages and economy rate analysis

- **Powerful Prediction Models**
  - Random Forest regressors with optimized hyperparameters
  - Separate models for batting and bowling performance
  - Confidence intervals from tree variance

- **Actionable Outputs**
  - Individual player performance predictions
  - Top performers analysis
  - CSV export capabilities

## 📊 Results

Our Random Forest models significantly outperform baseline linear regression:

| Metric          | Batting (R²) | Bowling (R²) |
|-----------------|-------------|-------------|
| Random Forest   | 0.82        | 0.78        |
| Linear Regression | 0.58        | 0.52        |

Key predictive features:
- **Batsmen**: Consistency (28.3%), Recent Form (22.1%), Boundary Percentage (18.7%)
- **Bowlers**: Economy Consistency (31.2%), Wicket Form (23.8%), Strike Rate (17.5%)
