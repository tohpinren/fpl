# FPL

A Python script to download all available Fantasy Premier League (FPL) API data for the current season.

[Open the Touchline dashboard](https://tohpinren.github.io/fpl/)

Updated for **2026/27**. The official API always serves the active season; players,
positions, teams, fixtures, points and bonus statistics come from that data rather
than a hardcoded season roster. See the [official 2026/27 changes](https://www.premierleague.com/en/news/4679873).

## Features

- Downloads bootstrap-static data (players, teams, gameweeks, events, etc.)
- Downloads all fixtures
- Downloads detailed player summaries for all players
- Downloads gameweek live data
- Organizes data in a structured directory format
- Includes retry logic and error handling
- Detects season changes and archives the previous season before refreshing reused player IDs

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

### GitHub Pages

The dashboard can be published as a static site using
[`.github/workflows/pages.yml`](.github/workflows/pages.yml). Set the repository's
**Settings → Pages → Source** to **GitHub Actions**. The workflow deploys on pushes
to `main`, on manual runs, and twice daily at 08:17 and 20:17 Singapore time
(GitHub may delay scheduled runs).

Each deployment downloads current FPL data, runs the existing position analysis,
and exports only the frontend and `dashboard.json`. Raw player files and season
archives are excluded. An incomplete download fails the deployment and leaves
the previous published site available.

On the hosted site, **Refresh view** reloads the latest published results. Search,
filters, player details, the fixture planner and watchlists run in the browser.
Watchlists are stored on the visitor's browser, separately for each season.

Build and preview the static version locally:

```bash
uv run build_site.py --refresh
python -m http.server 8080 --directory dist
```

Omit `--refresh` to build from existing local downloads. GitHub Pages needs a
public repository on GitHub Free, or a plan that supports Pages for private
repositories.

### Web dashboard

Open the local transfer dashboard with the same Python environment as the analysis scripts:

```bash
uv run serve.py
# Or: python serve.py
```

Visit `http://127.0.0.1:8000`. Use `--port 8080` to choose another port.
The dashboard includes position rankings, player and club search, price filters,
sorting, a browser-saved watchlist, player details and an upcoming fixture planner.
**Refresh data** downloads the current FPL season through the existing downloader.
If no local data exists yet, use that button to get started.

Rankings reuse the existing Python analysis without training ML models on page
loads. Scores are compared within a position; fixture colours use official FPL
difficulty ratings. The frontend uses plain HTML, CSS and JavaScript, and the
local server uses Python's standard library. No frontend install or build step
is needed. The server listens on loopback for personal, local use.

### Download data from the command line

Run the download script:

```bash
python download_fpl_data.py
```

Run this before analysis to refresh the active season. The script downloads all
players and fixtures, plus live data for completed and current gameweeks. Future
gameweeks are skipped until they start.

When the opening gameweek deadline identifies a new season, existing data is moved
to `data/archive/<previous-season>/` before the new bootstrap data is saved. This
preserves your old downloads and prevents previous-season player histories from
being used when the API reuses player IDs. For example, upgrading from 2025/26
preserves the old files in `data/archive/2025-26/`.

The active data stays in `data/`, so existing analysis commands keep working:

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
├── archive/                   # Previous seasons, preserved on season changes
│   └── 2025-26/
├── leagues/                   # League standings (if downloaded)
└── teams/                     # Team data (if downloaded)
```

## API Endpoints Used

- `/bootstrap-static/` - General game data
- `/fixtures/` - All fixtures
- `/element-summary/{player_id}/` - Detailed player data
- `/event/{gameweek}/live/` - Live gameweek data

## Analysis Scripts

All analysis scripts use the refreshed current-season data:

```bash
python find_defenders.py --max-price 6.0 --lookahead-gws 5
python find_midfielders.py --max-price 8.5 --lookahead-gws 5
python find_forwards.py --max-price 9.0
```

Defensive contribution points follow the [official FPL rules](https://fantasy.premierleague.com/help/faqs):
defenders earn two points for at least 10 contributions in a match; midfielders
and forwards need 12. Points are capped at two per match. Rankings and prediction
features count qualifying matches, rather than estimating points from season
contribution totals. Midfielder and defender output includes the percentage of
appearances meeting the relevant threshold.

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
- `--lookahead-gws`: Number of upcoming gameweeks for midfielder and defender analysis (default: 5)

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- The script includes rate limiting to avoid overwhelming the API
- Data is saved as JSON files for easy analysis
- The `data/` directory is excluded from version control (see `.gitignore`)
- The ML model uses 49 features and is optimized for high-dimensional tabular data
- Model automatically selects the best performing algorithm (typically XGBoost or LightGBM)
