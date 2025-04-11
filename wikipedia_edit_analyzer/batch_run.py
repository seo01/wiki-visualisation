import argparse
import subprocess
import sys
import os
import re
import json
import time
from pathlib import Path

# Define directories relative to this script
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MAIN_SCRIPT_PATH = BASE_DIR / "main.py"

def generate_file_prefix(title):
    """Generates a safe filename prefix from an article title."""
    # Lowercase, replace spaces/special chars with underscore, limit length
    safe_title = re.sub(r'[^\w\s-]', '', title.lower())
    safe_title = re.sub(r'[-\s]+', '_', safe_title).strip('_')
    return safe_title[:50] # Limit prefix length

def run_pipeline_for_article(title, file_prefix, common_args):
    """Runs the main.py pipeline for a single article."""
    command = [
        sys.executable, str(MAIN_SCRIPT_PATH), title,
        "--file-prefix", file_prefix
    ]
    # Pass through common args like --num_revisions, --num_predictions, --look_back
    command.extend(common_args)

    print(f"\n=== Processing article: '{title}' (Prefix: '{file_prefix}') ===")
    print(f"Running command: {' '.join(command)}")
    
    start_time = time.time()
    try:
        # Use check=True to raise error on failure
        subprocess.run(command, check=True, text=True, cwd=BASE_DIR)
        end_time = time.time()
        print(f"--- Successfully processed '{title}' in {end_time - start_time:.2f} seconds ---")
        return True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"--- Error processing '{title}' (Failed with code {e.returncode} after {end_time - start_time:.2f} seconds) ---", file=sys.stderr)
        return False
    except Exception as e:
        end_time = time.time()
        print(f"--- Unexpected error processing '{title}' after {end_time - start_time:.2f} seconds: {e} ---", file=sys.stderr)
        return False

def read_score(file_prefix):
    """Reads the prediction score from the JSON file."""
    score_filename = f"{file_prefix}_score.json"
    score_path = OUTPUT_DIR / score_filename
    try:
        with open(score_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Use .get() for safer access in case key is missing
            score = data.get("prediction_error_distance_2d") 
            if score is not None:
                return float(score)
            else:
                 print(f"Warning: Key 'prediction_error_distance_2d' not found in {score_path}", file=sys.stderr)
                 return float('inf') # Assign high score if key missing
    except FileNotFoundError:
        print(f"Warning: Score file not found: {score_path}", file=sys.stderr)
        return float('inf') # Assign high score if file missing
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"Warning: Error reading or parsing score file {score_path}: {e}", file=sys.stderr)
        return float('inf') # Assign high score on error

def generate_summary_html(results, output_html_path):
    """Generates a summary HTML page with iframes for each result, sorted by score."""
    # Sort results by score (ascending), failures (inf score) will be at the end
    results.sort(key=lambda x: x['score'])

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wikipedia Edit Path Batch Summary</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1 { text-align: center; margin-bottom: 30px; }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); /* Responsive 2 columns */
            gap: 30px;
        }
        .grid-item {
            border: 1px solid #ccc;
            padding: 15px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            display: flex; /* Use flexbox for internal layout */
            flex-direction: column;
        }
        .grid-item h2 { margin-top: 0; font-size: 1.2em; }
        .grid-item iframe {
            border: none;
            width: 100%;
            height: 500px; /* Adjust height as needed */
            flex-grow: 1; /* Allow iframe to fill available space */
        }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>Wikipedia Edit Path Batch Summary</h1>
    <p>Results ordered by prediction error (lower is better).</p>
    <div class="grid-container">
"""

    for result in results:
        title = result['title']
        score = result['score']
        html_path = result['html_path']
        prefix = result['prefix']
        
        # Use relative path for iframe src if possible
        try:
             iframe_src = os.path.relpath(OUTPUT_DIR / html_path, start=OUTPUT_DIR)
        except ValueError:
             iframe_src = html_path # Fallback if relative path fails

        html_content += f'        <div class="grid-item">\n'
        if score == float('inf'):
             html_content += f'            <h2>{title} <span class="error">(Failed or Score Missing)</span></h2>\n'
        else:
             html_content += f'            <h2>{title} (Score: {score:.3f})</h2>\n'
        
        # Check if the individual HTML file exists before adding iframe
        if os.path.exists(OUTPUT_DIR / html_path):
            html_content += f'            <iframe src="{iframe_src}"></iframe>\n'
        else:
             html_content += f'            <p class="error">Plot HTML file not found: {html_path}</p>\n'

        html_content += f'        </div>\n'

    html_content += """
    </div>
</body>
</html>
"""

    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nBatch summary HTML generated successfully at {output_html_path}")
    except IOError as e:
        print(f"Error writing summary HTML to {output_html_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Run the Wikipedia edit analysis pipeline for multiple articles.")
    parser.add_argument("-i", "--input-file", required=True, help="Path to a text file containing Wikipedia article titles, one per line.")
    # Add common args to pass through to main.py
    parser.add_argument("--num_revisions", type=int, default=100, help="Number of revisions to fetch (default: 100).")
    parser.add_argument("--num_predictions", type=int, default=10, help="Number of future edits to predict (default: 10).")
    parser.add_argument("--look_back", type=int, default=10, help="Number of recent edits for prediction trend (default: 10).")

    args = parser.parse_args()

    # Prepare common arguments to pass to main.py
    common_pipeline_args = [
         "--num_revisions", str(args.num_revisions),
         "--num_predictions", str(args.num_predictions),
         "--look_back", str(args.look_back)
    ]

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            titles = [line.strip() for line in f if line.strip()]
        if not titles:
             print(f"Error: Input file '{args.input_file}' is empty or contains no valid titles.", file=sys.stderr)
             sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file '{args.input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    total_articles = len(titles)
    print(f"Starting batch processing for {total_articles} articles...")

    for i, title in enumerate(titles):
        print(f"\n[{i+1}/{total_articles}] Processing '{title}'...")
        file_prefix = generate_file_prefix(title)
        success = run_pipeline_for_article(title, file_prefix, common_pipeline_args)
        
        score = float('inf') # Default score for failure
        if success:
             score = read_score(file_prefix)
             
        results.append({
            "title": title,
            "prefix": file_prefix,
            "score": score,
            "html_path": f"{file_prefix}_edit_path.html",
            "success": success
        })

    print("\n=== Batch Processing Complete ===")
    successful_runs = sum(1 for r in results if r['success'])
    print(f"Successfully processed {successful_runs} out of {total_articles} articles.")

    # Generate summary HTML
    summary_output_path = OUTPUT_DIR / "batch_summary.html"
    generate_summary_html(results, summary_output_path)


if __name__ == "__main__":
    main()
