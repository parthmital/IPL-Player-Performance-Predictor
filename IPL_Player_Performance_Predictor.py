import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IPLPerformanceAnalyzer:
    """
    Analyzes and predicts IPL player performance based on historical data.
    
    This class provides methods to analyze batsman and bowler performance metrics
    and predict future performance using machine learning models.
    """
    
    def __init__(self, file_path=None, data=None):
        """
        Initialize the IPL Performance Analyzer.
        
        Args:
            file_path (str, optional): Path to the CSV file containing IPL match data.
            data (DataFrame, optional): Pre-loaded DataFrame containing IPL match data.
        
        At least one of file_path or data must be provided.
        """
        self.file_path = file_path
        self.df = data
        self.batsman_stats = None
        self.bowler_stats = None
        self.batsman_model = None
        self.bowler_model = None
        self.batsman_scaler = StandardScaler()
        self.bowler_scaler = StandardScaler()
        
        if file_path is None and data is None:
            raise ValueError("Either file_path or data must be provided")
            
        if data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
            
    def load_and_preprocess(self):
        """
        Load data from CSV file and preprocess it for analysis.
        
        This method optimizes data loading with appropriate dtypes and
        handles missing values in both categorical and numerical columns.
        
        Returns:
            self: Returns the instance itself for method chaining.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the loaded data doesn't have required columns.
        """
        if self.df is not None:
            logger.info("Using pre-loaded DataFrame")
            return self
            
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
                
            logger.info(f"Loading data from {self.file_path}")
            
            # Define column dtypes for optimized loading
            dtypes = {
                'match_id': 'int32',
                'inning': 'int8',
                'over': 'int8',
                'ball': 'int8',
                'batsman_runs': 'int8',
                'extra_runs': 'int8',
                'total_runs': 'int8',
                'is_wicket': 'int8'
            }
            
            # Read CSV with optimized parameters
            self.df = pd.read_csv(
                self.file_path, 
                dtype=dtypes,
                low_memory=True
            )
            
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise e
            else:
                raise ValueError(f"Error loading data: {str(e)}")
        
        # Validate required columns
        required_columns = ['match_id', 'batsman_runs', 'batter', 'bowler', 'is_wicket']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
        # Define categorical columns and their default values
        categorical_defaults = {
            'player_dismissed': 'not_dismissed',
            'dismissal_kind': 'not_dismissed',
            'fielder': 'no_fielder',
            'extras_type': 'no_extras',
            'batting_team': 'unknown',
            'bowling_team': 'unknown',
            'batter': 'unknown',
            'bowler': 'unknown'
        }
        
        # Handle categorical columns
        for col, default_val in categorical_defaults.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(default_val).astype('category')
        
        # Fill numerical NA values with 0
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        self.df[numerical_cols] = self.df[numerical_cols].fillna(0)
        
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        
        return self

    def _generate_batsman_stats(self):
        """
        Generate comprehensive batting statistics for all batsmen.
        
        This method calculates various batting metrics including total runs,
        strike rate, average, and consistency using vectorized operations.
        
        Returns:
            DataFrame: A DataFrame containing batsman statistics.
        
        Raises:
            ValueError: If data hasn't been loaded properly.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
            
        logger.info("Generating batsman statistics")
        
        # Create aggregation dictionary for batsman metrics
        agg_dict = {
            'batsman_runs': ['sum', lambda x: (x == 4).sum(), lambda x: (x == 6).sum()],
            'ball': 'count',
            'match_id': 'nunique',
            'is_wicket': 'sum'
        }
        
        # Group by batsman and calculate basic statistics
        batsman_stats = self.df.groupby('batter').agg(agg_dict)
        batsman_stats.columns = ['total_runs', 'fours', 'sixes', 'balls_faced', 'matches_played', 'dismissals']
        batsman_stats = batsman_stats.reset_index()
        
        # Filter out players with insufficient data (at least 10 balls faced)
        batsman_stats = batsman_stats[batsman_stats['balls_faced'] >= 10]
        
        # Calculate derived metrics
        batsman_stats['strike_rate'] = (batsman_stats['total_runs'] / batsman_stats['balls_faced']) * 100
        batsman_stats['average'] = batsman_stats['total_runs'] / np.maximum(batsman_stats['dismissals'], 1)
        batsman_stats['boundary_percentage'] = ((batsman_stats['fours'] + batsman_stats['sixes']) / 
                                              np.maximum(batsman_stats['balls_faced'], 1)) * 100
        
        # Calculate median runs per match for improved consistency measure
        runs_per_match = self.df.groupby(['batter', 'match_id'])['batsman_runs'].sum().reset_index()
        
        # Vectorized consistency calculation using coefficient of variation
        consistency_std = runs_per_match.groupby('batter')['batsman_runs'].std().fillna(0)
        consistency_mean = runs_per_match.groupby('batter')['batsman_runs'].mean()
        consistency_cv = (consistency_std / np.maximum(consistency_mean, 1))
        
        # Invert so higher values mean more consistent
        consistency = (1 - consistency_cv).clip(0, 1).reset_index()
        consistency.columns = ['batter', 'consistency']
        
        # Calculate form (recent performance trend)
        recent_matches = self.df.sort_values('match_id', ascending=False)
        recent_perfomance = recent_matches.groupby(['batter', 'match_id'])['batsman_runs'].sum().reset_index()
        
        # Get last 5 matches (if available) for each player
        form_data = []
        for player in batsman_stats['batter'].unique():
            player_recent = recent_perfomance[recent_perfomance['batter'] == player].head(5)
            if not player_recent.empty:
                weighted_form = np.average(
                    player_recent['batsman_runs'], 
                    weights=np.linspace(1, 0.6, len(player_recent))
                )
                form_data.append({'batter': player, 'form': weighted_form})
                
        form_df = pd.DataFrame(form_data)
        
        # Merge all statistics
        batsman_stats = pd.merge(batsman_stats, consistency, on='batter', how='left')
        batsman_stats = pd.merge(batsman_stats, form_df, on='batter', how='left')
        batsman_stats['form'] = batsman_stats['form'].fillna(batsman_stats['total_runs'] / batsman_stats['matches_played'])
        
        # Store for later use
        self.batsman_stats = batsman_stats
        
        logger.info(f"Generated statistics for {len(batsman_stats)} batsmen")
        
        return self.batsman_stats

    def _generate_bowler_stats(self):
        """
        Generate comprehensive bowling statistics for all bowlers.
        
        This method calculates various bowling metrics including wickets,
        economy rate, average, and consistency using vectorized operations.
        
        Returns:
            DataFrame: A DataFrame containing bowler statistics.
            
        Raises:
            ValueError: If data hasn't been loaded properly.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
            
        logger.info("Generating bowler statistics")
        
        # Create aggregation dictionary for bowler metrics
        agg_dict = {
            'is_wicket': 'sum',
            'ball': 'count',
            'total_runs': 'sum',
            'match_id': 'nunique',
            'extra_runs': 'sum'
        }
        
        # Group by bowler and calculate basic statistics
        bowler_stats = self.df.groupby('bowler').agg(agg_dict)
        bowler_stats.columns = ['wickets', 'balls_bowled', 'runs_conceded', 'matches_played', 'extras']
        bowler_stats = bowler_stats.reset_index()
        
        # Filter out players with insufficient data (at least 30 balls bowled)
        bowler_stats = bowler_stats[bowler_stats['balls_bowled'] >= 30]
        
        # Calculate derived metrics
        bowler_stats['overs'] = bowler_stats['balls_bowled'] / 6
        bowler_stats['economy'] = bowler_stats['runs_conceded'] / np.maximum(bowler_stats['overs'], 1)
        bowler_stats['average'] = bowler_stats['runs_conceded'] / np.maximum(bowler_stats['wickets'], 1)
        bowler_stats['strike_rate'] = bowler_stats['balls_bowled'] / np.maximum(bowler_stats['wickets'], 1)
        
        # Calculate wickets per match
        bowler_stats['wickets_per_match'] = bowler_stats['wickets'] / bowler_stats['matches_played']
        
        # Vectorized economy consistency calculation using coefficient of variation
        economy_per_match = (self.df.groupby(['bowler', 'match_id'])
                           .apply(lambda x: x['total_runs'].sum() / np.maximum(x['ball'].count() / 6, 1))
                           .reset_index(name='economy'))
        
        # Calculate consistency as inverse of coefficient of variation
        eco_std = economy_per_match.groupby('bowler')['economy'].std().fillna(0)
        eco_mean = economy_per_match.groupby('bowler')['economy'].mean()
        eco_cv = (eco_std / np.maximum(eco_mean, 1))
        
        # Invert so higher values mean more consistent
        consistency = (1 - eco_cv).clip(0, 1).reset_index()
        consistency.columns = ['bowler', 'consistency']
        
        # Calculate form (recent performance trend)
        recent_matches = self.df.sort_values('match_id', ascending=False)
        wickets_per_match = recent_matches.groupby(['bowler', 'match_id'])['is_wicket'].sum().reset_index()
        
        # Get last 5 matches (if available) for each player
        form_data = []
        for player in bowler_stats['bowler'].unique():
            player_recent = wickets_per_match[wickets_per_match['bowler'] == player].head(5)
            if not player_recent.empty:
                weighted_form = np.average(
                    player_recent['is_wicket'], 
                    weights=np.linspace(1, 0.6, len(player_recent))
                )
                form_data.append({'bowler': player, 'form': weighted_form})
                
        form_df = pd.DataFrame(form_data)
        
        # Merge all statistics
        bowler_stats = pd.merge(bowler_stats, consistency, on='bowler', how='left')
        bowler_stats = pd.merge(bowler_stats, form_df, on='bowler', how='left')
        bowler_stats['form'] = bowler_stats['form'].fillna(bowler_stats['wickets'] / bowler_stats['matches_played'])
        
        # Store for later use
        self.bowler_stats = bowler_stats
        
        logger.info(f"Generated statistics for {len(bowler_stats)} bowlers")
        
        return self.bowler_stats

    def _train_batsman_model(self):
        """
        Train a machine learning model to predict batsman performance.
        
        This method uses RandomForestRegressor to predict runs based on
        various batsman statistics and incorporates cross-validation.
        
        Returns:
            self: Returns the instance itself for method chaining.
            
        Raises:
            ValueError: If batsman statistics have not been generated.
        """
        if self.batsman_stats is None:
            self._generate_batsman_stats()
            
        if len(self.batsman_stats) < 10:
            raise ValueError("Insufficient data to train model (fewer than 10 batsmen)")
            
        logger.info("Training batsman performance model")
        
        # Prepare features and target
        feature_cols = ['fours', 'sixes', 'balls_faced', 'matches_played', 'dismissals', 
                       'strike_rate', 'average', 'boundary_percentage', 'consistency', 'form']
        
        # Check for available columns
        available_features = [col for col in feature_cols if col in self.batsman_stats.columns]
        
        features = self.batsman_stats[available_features].copy()
        target = self.batsman_stats['total_runs'].values
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features for better model performance
        X_train_scaled = self.batsman_scaler.fit_transform(X_train)
        X_val_scaled = self.batsman_scaler.transform(X_val)
        
        # Train model with hyperparameters
        self.batsman_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.batsman_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.batsman_model.score(X_train_scaled, y_train)
        val_score = self.batsman_model.score(X_val_scaled, y_val)
        
        logger.info(f"Batsman model R² scores - Train: {train_score:.4f}, Validation: {val_score:.4f}")
        
        # Store feature names for prediction
        self.batsman_feature_names = available_features
        
        return self

    def _train_bowler_model(self):
        """
        Train a machine learning model to predict bowler performance.
        
        This method uses RandomForestRegressor to predict wickets based on
        various bowler statistics and incorporates cross-validation.
        
        Returns:
            self: Returns the instance itself for method chaining.
            
        Raises:
            ValueError: If bowler statistics have not been generated.
        """
        if self.bowler_stats is None:
            self._generate_bowler_stats()
            
        if len(self.bowler_stats) < 10:
            raise ValueError("Insufficient data to train model (fewer than 10 bowlers)")
            
        logger.info("Training bowler performance model")
        
        # Prepare features and target
        feature_cols = ['balls_bowled', 'runs_conceded', 'matches_played', 'extras',
                       'overs', 'economy', 'average', 'strike_rate', 'consistency', 'form']
        
        # Check for available columns
        available_features = [col for col in feature_cols if col in self.bowler_stats.columns]
        
        features = self.bowler_stats[available_features].copy()
        target = self.bowler_stats['wickets'].values
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features for better model performance
        X_train_scaled = self.bowler_scaler.fit_transform(X_train)
        X_val_scaled = self.bowler_scaler.transform(X_val)
        
        # Train model with hyperparameters
        self.bowler_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bowler_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.bowler_model.score(X_train_scaled, y_train)
        val_score = self.bowler_model.score(X_val_scaled, y_val)
        
        logger.info(f"Bowler model R² scores - Train: {train_score:.4f}, Validation: {val_score:.4f}")
        
        # Store feature names for prediction
        self.bowler_feature_names = available_features
        
        return self

    def predict_batsman_performance(self, batsman_name, future_matches=5):
        """
        Predict performance for a single batsman.
        
        Args:
            batsman_name (str): Name of the batsman to predict for.
            future_matches (int, optional): Number of future matches to predict. Defaults to 5.
            
        Returns:
            dict: Dictionary containing prediction results.
            
        Raises:
            ValueError: If the batsman is not found in the data.
        """
        # Generate statistics and train model if needed
        if self.batsman_stats is None:
            self._generate_batsman_stats()
        if self.batsman_model is None:
            self._train_batsman_model()
            
        logger.info(f"Predicting performance for batsman: {batsman_name}")
        
        # Check if batsman exists in data
        if batsman_name not in self.batsman_stats['batter'].values:
            return {"error": f"Batsman {batsman_name} not found in the dataset"}
            
        # Get batsman data
        batsman_data = self.batsman_stats[self.batsman_stats['batter'] == batsman_name]
        matches = batsman_data['matches_played'].values[0]
        
        # Prepare features
        features = batsman_data[self.batsman_feature_names].copy()
        
        # Scale features
        features_scaled = self.batsman_scaler.transform(features)
        
        # Make prediction and normalize by matches
        predicted_runs = self.batsman_model.predict(features_scaled)[0] / matches * future_matches
        
        # Calculate confidence interval based on tree variance
        predictions = []
        for estimator in self.batsman_model.estimators_:
            pred = estimator.predict(features_scaled)[0] / matches * future_matches
            predictions.append(pred)
            
        lower_ci = np.percentile(predictions, 25)
        upper_ci = np.percentile(predictions, 75)
        
        # Create rich result dictionary
        result = {
            'batsman': batsman_name,
            'matches_analyzed': int(matches),
            'current_statistics': {
                'total_runs': int(batsman_data['total_runs'].values[0]),
                'average': round(batsman_data['average'].values[0], 2),
                'strike_rate': round(batsman_data['strike_rate'].values[0], 2),
                'boundary_percentage': round(batsman_data['boundary_percentage'].values[0], 2),
                'consistency_score': round(batsman_data['consistency'].values[0], 2),
            },
            'prediction': {
                'future_matches': future_matches,
                'predicted_runs': round(predicted_runs, 1),
                'predicted_average': round(predicted_runs / future_matches, 2),
                'confidence_interval': {
                    'lower': round(lower_ci, 1),
                    'upper': round(upper_ci, 1)
                }
            }
        }
        
        # Add form if available
        if 'form' in batsman_data.columns:
            result['current_statistics']['recent_form'] = round(batsman_data['form'].values[0], 2)
            
        return result

    def predict_bowler_performance(self, bowler_name, future_matches=5):
        """
        Predict performance for a single bowler.
        
        Args:
            bowler_name (str): Name of the bowler to predict for.
            future_matches (int, optional): Number of future matches to predict. Defaults to 5.
            
        Returns:
            dict: Dictionary containing prediction results.
            
        Raises:
            ValueError: If the bowler is not found in the data.
        """
        # Generate statistics and train model if needed
        if self.bowler_stats is None:
            self._generate_bowler_stats()
        if self.bowler_model is None:
            self._train_bowler_model()
            
        logger.info(f"Predicting performance for bowler: {bowler_name}")
        
        # Check if bowler exists in data
        if bowler_name not in self.bowler_stats['bowler'].values:
            return {"error": f"Bowler {bowler_name} not found in the dataset"}
            
        # Get bowler data
        bowler_data = self.bowler_stats[self.bowler_stats['bowler'] == bowler_name]
        matches = bowler_data['matches_played'].values[0]
        
        # Prepare features
        features = bowler_data[self.bowler_feature_names].copy()
        
        # Scale features
        features_scaled = self.bowler_scaler.transform(features)
        
        # Make prediction and normalize by matches
        predicted_wickets = self.bowler_model.predict(features_scaled)[0] / matches * future_matches
        
        # Calculate confidence interval based on tree variance
        predictions = []
        for estimator in self.bowler_model.estimators_:
            pred = estimator.predict(features_scaled)[0] / matches * future_matches
            predictions.append(pred)
            
        lower_ci = np.percentile(predictions, 25)
        upper_ci = np.percentile(predictions, 75)
        
        # Create rich result dictionary
        result = {
            'bowler': bowler_name,
            'matches_analyzed': int(matches),
            'current_statistics': {
                'total_wickets': int(bowler_data['wickets'].values[0]),
                'economy': round(bowler_data['economy'].values[0], 2),
                'average': round(bowler_data['average'].values[0], 2),
                'strike_rate': round(bowler_data['strike_rate'].values[0], 2),
                'consistency_score': round(bowler_data['consistency'].values[0], 2),
            },
            'prediction': {
                'future_matches': future_matches,
                'predicted_wickets': round(predicted_wickets, 1),
                'predicted_economy': round(bowler_data['economy'].values[0], 2),
                'confidence_interval': {
                    'lower': round(lower_ci, 1),
                    'upper': round(upper_ci, 1)
                }
            }
        }
        
        # Add form if available
        if 'form' in bowler_data.columns:
            result['current_statistics']['recent_form'] = round(bowler_data['form'].values[0], 2)
            
        return result
        
    def get_top_performers(self, category='batting', metric=None, top_n=10):
        """
        Get the top performers in a given category.
        
        Args:
            category (str): Either 'batting' or 'bowling'.
            metric (str, optional): Specific metric to sort by. If None, uses
                                   'total_runs' for batting and 'wickets' for bowling.
            top_n (int, optional): Number of top performers to return. Defaults to 10.
            
        Returns:
            DataFrame: Top performers sorted by the specified metric.
        """
        if category.lower() == 'batting':
            if self.batsman_stats is None:
                self._generate_batsman_stats()
                
            stats = self.batsman_stats.copy()
            if metric is None:
                metric = 'total_runs'
                
            # Ensure metric exists
            if metric not in stats.columns:
                raise ValueError(f"Metric '{metric}' not found in batsman statistics")
                
            # Get top performers
            top_players = stats.sort_values(metric, ascending=False).head(top_n)
            return top_players.loc[:, ['batter', metric] + [col for col in stats.columns if col != 'batter' and col != metric]]
            
        elif category.lower() == 'bowling':
            if self.bowler_stats is None:
                self._generate_bowler_stats()
                
            stats = self.bowler_stats.copy()
            if metric is None:
                metric = 'wickets'
                
            # Handle special cases for bowling metrics where lower is better
            ascending = metric in ['economy', 'average', 'strike_rate']
            
            # Ensure metric exists
            if metric not in stats.columns:
                raise ValueError(f"Metric '{metric}' not found in bowler statistics")
                
            # Get top performers
            top_players = stats.sort_values(metric, ascending=ascending).head(top_n)
            return top_players.loc[:, ['bowler', metric] + [col for col in stats.columns if col != 'bowler' and col != metric]]
            
        else:
            raise ValueError("Category must be either 'batting' or 'bowling'")

def export_all_players(file_path):
    """
    Export predictions for all players to CSV files.
    
    Args:
        file_path (str): Path to the IPL dataset CSV file.
        
    Returns:
        tuple: Paths to the exported CSV files.
    """
    try:
        logger.info("Initializing IPL Performance Analyzer")
        analyzer = IPLPerformanceAnalyzer(file_path)
        analyzer.load_and_preprocess()
        
        logger.info("Generating batsmen predictions")
        batsmen_stats = analyzer._generate_batsman_stats()
        analyzer._train_batsman_model()
        
        all_batsmen = []
        for batsman in batsmen_stats['batter'].unique():
            pred = analyzer.predict_batsman_performance(batsman, 5)
            if 'error' not in pred:
                all_batsmen.append(pred)
        
        batsmen_df = pd.json_normalize(all_batsmen)
        batsmen_output = "Batsmen_Predictions.csv"
        batsmen_df.to_csv(batsmen_output, index=False)
        
        logger.info("Generating bowlers predictions")
        bowlers_stats = analyzer._generate_bowler_stats()
        analyzer._train_bowler_model()
        
        all_bowlers = []
        for bowler in bowlers_stats['bowler'].unique():
            pred = analyzer.predict_bowler_performance(bowler, 5)
            if 'error' not in pred:
                all_bowlers.append(pred)
        
        bowlers_df = pd.json_normalize(all_bowlers)
        bowlers_output = "Bowlers_Predictions.csv"
        bowlers_df.to_csv(bowlers_output, index=False)
        
        logger.info(f"Successfully exported predictions for {len(all_batsmen)} batsmen and {len(all_bowlers)} bowlers")
        
        return batsmen_output, bowlers_output
        
    except Exception as e:
        logger.error(f"Error exporting predictions: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        file_path = "IPL Complete Dataset (2008-2024).csv"
        batsmen_file, bowlers_file = export_all_players(file_path)
        print(f"Exported batsmen predictions to: {batsmen_file}")
        print(f"Exported bowlers predictions to: {bowlers_file}")
    except Exception as e:
        print(f"Error: {str(e)}")