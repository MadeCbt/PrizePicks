import logging
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import modules
from src.data_fetch.espn_scraper import fetch_last_20_games
from src.data_fetch.prizepicks_fetch import fetch_betting_lines
from src.preprocessing.data_cleaner import clean_espn_data, merge_prizepicks_lines
from src.preprocessing.feature_engineer import create_features
from src.analysis.bet_generator import generate_bets
from src.analysis.ml_model import train_model, predict_bet_success


# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Fetching player data from ESPN...")
    espn_data = fetch_last_20_games()
    
    logging.info("Fetching PrizePicks betting lines...")
    prizepicks_data = fetch_betting_lines()
    
    logging.info("Cleaning and preprocessing data...")
    cleaned_espn_data = clean_espn_data(espn_data)
    combined_data = merge_prizepicks_lines(cleaned_espn_data, prizepicks_data)
    
    logging.info("Creating features for analysis...")
    feature_data = create_features(combined_data)
    
    logging.info("Generating bet combinations...")
    # Generate the top 10 2-man, 3-man, and 4-man bets with a minimum 30% chance of hitting
    two_man_bets = generate_bets(feature_data, bet_size=2, min_probability=0.3, top_n=10)
    three_man_bets = generate_bets(feature_data, bet_size=3, min_probability=0.3, top_n=10)
    four_man_bets = generate_bets(feature_data, bet_size=4, min_probability=0.3, top_n=10)

    # Debug print statements to verify the contents of the bets
    print("two_man_bets:", two_man_bets)
    print("three_man_bets:", three_man_bets)
    print("four_man_bets:", four_man_bets)
    
    logging.info("Training machine learning model on historical bets...")
    train_model()
    
    logging.info("Predicting success for new bets...")
    # Predict the probability of each bet combination hitting
    two_man_predictions = [predict_bet_success(bet) for bet in two_man_bets]
    three_man_predictions = [predict_bet_success(bet) for bet in three_man_bets]
    four_man_predictions = [predict_bet_success(bet) for bet in four_man_bets]
    
    # Collect results for output
    results = {
        "2-man Bets": list(zip(two_man_bets, two_man_predictions)),
        "3-man Bets": list(zip(three_man_bets, three_man_predictions)),
        "4-man Bets": list(zip(four_man_bets, four_man_predictions))
    }
    
    # Output results
    for bet_type, bets in results.items():
        logging.info(f"\nRecommended {bet_type} with Predictions:")
        for bet, prediction in bets:
            logging.info(f"Bet: {bet} - Prediction: {prediction}")

if __name__ == "__main__":
    main()
