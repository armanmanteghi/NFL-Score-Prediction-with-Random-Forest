import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
file_path = r'C:\Users\arman\OneDrive\Desktop\Coding\scores.csv'
data = pd.read_csv(file_path)

# Preprocess the data
data[['Visitor_Score', 'Home_Score']] = data['Score'].str.split('-', expand=True).astype(int)

# Create additional features
team_stats = {}
for _, row in data.iterrows():
    visitor, home = row['Visitor'], row['Home']
    visitor_score, home_score = row['Visitor_Score'], row['Home_Score']
    
    # Initialize stats if not present
    if visitor not in team_stats:
        team_stats[visitor] = {'Wins': 0, 'Losses': 0, 'Points_Scored': 0, 'Points_Allowed': 0}
    if home not in team_stats:
        team_stats[home] = {'Wins': 0, 'Losses': 0, 'Points_Scored': 0, 'Points_Allowed': 0}
    
    # Update stats
    if visitor_score > home_score:
        team_stats[visitor]['Wins'] += 1
        team_stats[home]['Losses'] += 1
    else:
        team_stats[home]['Wins'] += 1
        team_stats[visitor]['Losses'] += 1
    
    team_stats[visitor]['Points_Scored'] += visitor_score
    team_stats[visitor]['Points_Allowed'] += home_score
    team_stats[home]['Points_Scored'] += home_score
    team_stats[home]['Points_Allowed'] += visitor_score

# Convert team stats to DataFrame
stats_df = pd.DataFrame.from_dict(team_stats, orient='index').reset_index()
stats_df.rename(columns={'index': 'Team'}, inplace=True)

# Input variables for the teams
home_team = 'Miami Dolphins'  # Change this to your desired home team
visitor_team = 'Denver Broncos'  # Change this to your desired visitor team

# Prepare the dataset for a specific matchup
def prepare_data(team1, team2):
    team1_stats = stats_df[stats_df['Team'] == team1].iloc[0]
    team2_stats = stats_df[stats_df['Team'] == team2].iloc[0]
    
    # Create features for both teams
    features = [
        team1_stats['Wins'], team1_stats['Losses'], team1_stats['Points_Scored'], team1_stats['Points_Allowed'],
        team2_stats['Wins'], team2_stats['Losses'], team2_stats['Points_Scored'], team2_stats['Points_Allowed']
    ]
    
    return features

# Prepare dataset
features = []
labels = []

# Create a feature set for each matchup in the original data
for _, row in data.iterrows():
    features.append(prepare_data(row['Visitor'], row['Home']))
    labels.append(row['Home_Score'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Predict a future game
future_game_features = prepare_data(visitor_team, home_team)

# Predict scores for both teams
predicted_score_home = model.predict([future_game_features])[0]  # Score for the home team

# Predict for the reverse matchup to get the away team score
predicted_score_away = model.predict([prepare_data(home_team, visitor_team)])[0]

# Output the predicted scores
print(f'Predicted score for {visitor_team}: {predicted_score_away}')
print(f'Predicted score for {home_team}: {predicted_score_home}')
