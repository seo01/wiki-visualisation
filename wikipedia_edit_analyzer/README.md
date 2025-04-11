# Wikipedia Edit Analyzer

This project analyzes the revision history of a Wikipedia article, visualizes the 'path' of edits in a 2D space, and attempts to predict future edits.

## Features

*   Fetches the content of recent revisions for a given Wikipedia article (currently uses placeholder content).
*   Converts revision texts into numerical vectors using TF-IDF.
*   Predicts future edit vectors using simple linear extrapolation based on recent trends.
*   Calculates the differences (diffs) between consecutive revisions.
*   Reduces the dimensionality of historical and predicted vectors using UMAP.
*   Generates an **interactive HTML 2D plot** visualizing:
    *   The historical edit path up to `N-M` edits ago (where `N` is total revisions fetched and `M` is `--num_predictions`).
    *   A **fork** showing both the *actual* path for the last `M` edits and the *predicted* path for those same `M` edits, allowing visual comparison.
    *   Diffs between consecutive historical revisions are shown on hover.
    *   An annotation showing the Euclidean distance between the final actual point and the final predicted point in the 2D UMAP space (a simple prediction error score).

## Project Structure

```
wikipedia_edit_analyzer/
├── data/                 # Stores intermediate data (revisions, vectors)
│   ├── raw_edits.json
│   ├── edit_vectors.npy
│   ├── edit_vectors.npy
│   ├── predicted_vectors.npy
│   └── edit_diffs.json       # Added: Diffs between revisions
├── output/               # Stores the final visualization
│   └── edit_path.html        # Changed: Output is now HTML
├── download_edits.py     # Step 1: Fetch edits
├── vectorize_edits.py    # Step 2: Convert edits to vectors (TF-IDF)
├── calculate_diffs.py    # Step 3: Calculate diffs between revisions
├── predict_edits.py      # Step 4: Predict future edit vectors (Extrapolation)
├── visualize_edits.py    # Step 5: Reduce dimensions (UMAP) and plot (Plotly HTML)
├── main.py               # Wrapper script to run the full pipeline
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

1.  **Clone the repository (if applicable) or ensure you have the project files.**
2.  **Navigate to the project directory:**
    ```bash
    cd path/to/wikipedia_edit_analyzer
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This now includes `plotly` and `requests`, and `matplotlib` is no longer strictly required by the main pipeline)*

## Usage

Run the main pipeline script from the `wikipedia_edit_analyzer` directory, providing the title of the Wikipedia article as an argument:

```bash
python main.py "Name of Wikipedia Article"
```

**Example:**

```bash
python main.py "Python (programming language)"
```
This will generate an interactive HTML file named `output/edit_path.html`. Open this file in a web browser. The plot shows the main historical path in grey, the actual final edits in green, and the predicted path in dashed orange, branching off from the main path. Hover over points on the grey and green paths to see the corresponding diff text.

### Options

You can customize the process using command-line arguments for `main.py`:

*   `--num_revisions N`: Specify the number of historical revisions to fetch (default: 100).
*   `--num_predictions N`: Specify the number of future edits to predict (default: 10).
*   `--look_back N`: Specify how many recent edits to use for the prediction trend (default: 10).

**Example with options:**

```bash
python main.py "Artificial intelligence" --num_revisions 50 --num_predictions 5 --look_back 5
```

### Individual Steps

You can also run individual steps:

```bash
# Download (Needs article title)
python download_edits.py "Article Title" -n 100 -o data

# Vectorize (Uses data/raw_edits.json by default)
python vectorize_edits.py -i data -o data

# Predict (Uses data/edit_vectors.npy by default)
python predict_edits.py -n 10 -l 10 -i data -o data

# Calculate Diffs (Uses data/raw_edits.json by default)
python calculate_diffs.py -i data -o data

# Predict (Uses data/edit_vectors.npy by default)
python predict_edits.py -n 10 -l 10 -i data -o data

# Visualize (Uses data files by default, prefix needed if not running via main.py)
python visualize_edits.py --file-prefix "your_prefix" -i_hist data -i_pred data -i_diffs data -o output -t "Custom Plot Title"
```

## Batch Processing

To process multiple articles at once and generate a summary page:

1.  **Create an input file** (e.g., `titles.txt`) listing the Wikipedia article titles you want to process, one title per line. Example `titles.txt`:
    ```text
    Python (programming language)
    JavaScript
    Artificial intelligence
    World War II
    Quantum computing
    Black hole
    Photosynthesis
    Sonnet
    Impressionism
    Game theory
    DNA
    Special relativity
    Machine learning
    Climate change
    COVID-19 pandemic
    Renaissance
    Internet
    Economics
    Philosophy
    Democracy
    String theory
    Graph theory
    Napoleon
    Alexander the Great
    Periodic table
    Calculus
    Evolution
    Cell (biology)
    Galaxy
    Solar System
    ```

2.  **Run the `batch_run.py` script**, providing the path to your input file:
    ```bash
    python batch_run.py --input-file titles.txt
    ```
    You can also pass pipeline arguments like `--num_revisions`, `--num_predictions`, `--look_back` to the batch script, and they will be used for all articles in the batch.
    ```bash
    # Example with custom parameters for a batch of 30 titles
    python batch_run.py --input-file titles.txt --num_revisions 50 --num_predictions 5
    ```

3.  **View the results:**
    *   Intermediate files (`_raw_edits.json`, `_edit_vectors.npy`, etc.) and individual outputs (`_edit_path.html`, `_score.json`) for each article will be saved in the `data/` and `output/` directories, prefixed with a slug generated from the article title.
    *   A summary HTML page `output/batch_summary.html` will be created, displaying the individual plots in a two-column layout, ordered by prediction score (lowest first).

## Notes

*   The `download_edits.py` script uses the MediaWiki API to fetch actual revisions. Ensure you respect Wikipedia's API usage policies (e.g., rate limits).
*   The prediction method is a simple linear extrapolation. More sophisticated methods (like LSTMs) could be implemented for potentially better results.
*   The prediction starts based on the trend derived from the first `N-M` revisions, where `N = --num_revisions` and `M = --num_predictions`.
*   The hover text displays the output of `difflib.unified_diff`. Very large diffs might be slow to render or cumbersome to read in the hover box.
