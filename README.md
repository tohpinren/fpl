# FPL

A Python script to download all available Fantasy Premier League (FPL) API data for the current season.

## Features

- Downloads bootstrap-static data (players, teams, gameweeks, events, etc.)
- Downloads all fixtures
- Downloads detailed player summaries for all players
- Downloads gameweek live data
- Organizes data in a structured directory format
- Includes retry logic and error handling

## Installation

### Using uv (Recommended)

1. Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:

```bash
uv sync --no-install-project
```

This will create a virtual environment and install all dependencies without trying to install the project itself. To run the script:

```bash
uv run download_fpl_data.py
```

Or activate the virtual environment:

```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
python download_fpl_data.py
```

### Using pip (Alternative)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

### Common uv Commands

- `uv sync --no-install-project` - Install dependencies without installing the project (for script-only projects)
- `uv sync` - Install dependencies and create/update lock file (requires buildable package)
- `uv run <script>` - Run a script in the virtual environment
- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency
- `uv lock` - Update the lock file without installing
- `uv pip install <package>` - Install a package directly (alternative to `uv add`)

## Usage

Run the download script:

```bash
python download_fpl_data.py
```

The script will download all available data and save it to the `data/` directory with the following structure:

```
data/
├── bootstrap-static.json      # Main data file with players, teams, gameweeks
├── fixtures.json              # All fixture information
├── players/                   # Individual player summaries
│   ├── 1.json
│   ├── 2.json
│   └── ...
├── gameweeks/                 # Live gameweek data
│   ├── 1.json
│   ├── 2.json
│   └── ...
├── leagues/                   # League standings (if downloaded)
└── teams/                     # Team data (if downloaded)
```

## API Endpoints Used

- `/bootstrap-static/` - General game data
- `/fixtures/` - All fixtures
- `/element-summary/{player_id}/` - Detailed player data
- `/event/{gameweek}/live/` - Live gameweek data

## Analysis Scripts

### Find Midfielders (`find_midfielders.py`)

Analyzes all midfielders to find the best transfer targets based on data-driven metrics:

- **Expected Goal Involvement (xGI)** per 90 minutes
- **Fixture Difficulty Rating (FDR)** for upcoming games
- **Recent form** and underlying stats trends
- **Rotation risk** and minutes played
- **Value score** (xGI per 90 per £million)
- **ICT Index** (Influence, Creativity, Threat)

Run the analysis:

```bash
# Show all midfielders
python find_midfielders.py

# Filter by maximum price (e.g., only players £8.5m or less)
python find_midfielders.py --max-price 8.5
```

The script ranks midfielders by a composite score that combines:

- xGI per 90 (weighted 3.0)
- Fixture difficulty (easier fixtures = higher score)
- Recent form
- Value for money
- Rotation risk (lower risk = higher score)
- **ML Predicted Points** (weighted 0.15) - Machine learning prediction of points in next 5 games

**Machine Learning Integration:**

The script automatically trains a prediction model using historical data and includes predicted points as a metric:

- Trains multiple models (Random Forest, XGBoost, LightGBM, Ridge) and selects the best
- Uses all available historical gameweeks for training
- Predicts total points for next 5 games for each midfielder
- Shows model performance metrics during training
- Displays predicted points alongside other metrics in the output
- Optimized hyperparameters for high-dimensional feature space (49 features)

**Comprehensive Feature Set (49 features):**

**Player Performance:**

- Expected metrics: xGI, xG, xA (per 90 and recent averages)
- Recent form: Points, goals, assists, bonus points (3, 5, and 10 game windows)
- ICT breakdown: Influence, Creativity, Threat (separate metrics)
- Season totals: Goals, assists, bonus points, minutes, starts

**Team & Fixture Context:**

- Team strength and form
- Fixture difficulty (FDR) based on current league standings
- Home/away performance splits
- Percentage of upcoming home fixtures

**Availability & Rotation:**

- Rotation risk metrics
- Recent minutes and start percentages
- Games with 60+ minutes played

**Market & Value:**

- Cost and value metrics (value_form, value_season)
- Ownership percentage
- Transfer trends (in/out ratios)
- FPL expected points (ep_next, ep_this)

**Disciplinary:**

- Yellow and red cards

## Command Line Options

- `--max-price`: Filter results by maximum price in millions (e.g., `--max-price 8.5` for £8.5m maximum)

## Notes

- The script includes rate limiting to avoid overwhelming the API
- Data is saved as JSON files for easy analysis
- The `data/` directory is excluded from version control (see `.gitignore`)
- The ML model uses 49 features and is optimized for high-dimensional tabular data
- Model automatically selects the best performing algorithm (typically XGBoost or LightGBM)
