import argparse
import subprocess
import sys
import os

# Define the paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure data and output directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_step(script_name, article_title, file_prefix, args_list=None):
    """Runs a step of the process using a subprocess, passing the file prefix."""
    script_path = os.path.join(BASE_DIR, script_name)
    command = [sys.executable, script_path]

    # Always add file prefix if provided
    if file_prefix:
        command.extend(["--file-prefix", file_prefix])

    # Add arguments selectively based on the script being run
    if script_name == "download_edits.py":
        # download_edits.py takes article_title directly + specific args below
        command.append(article_title)
        # Specific args like -n, --output_dir are added from args_list below
    elif script_name == "vectorize_edits.py":
        # Needs input/output dirs (which default to DATA_DIR in the script itself)
        command.extend(["--input_dir", DATA_DIR, "--output_dir", DATA_DIR])
    elif script_name == "calculate_diffs.py":
         # Needs input/output dirs (which default to DATA_DIR in the script itself)
         command.extend(["--input_dir", DATA_DIR, "--output_dir", DATA_DIR])
    elif script_name == "predict_edits.py":
        # Needs input/output dirs (which default to DATA_DIR in the script itself)
        command.extend(["--input_dir", DATA_DIR, "--output_dir", DATA_DIR])
        # Specific args like -n, -l are added from args_list below
    elif script_name == "visualize_edits.py":
        # Needs multiple input dirs and output dir
        command.extend([
            "--input_hist_dir", DATA_DIR,
            "--input_pred_dir", DATA_DIR,
            # Removed --input_diffs_dir
            "--output_dir", OUTPUT_DIR
        ])
        # Add plot title based on article title
        plot_title = f"Edit Path for '{article_title}'"
        command.extend(["--title", plot_title])
        # Specific args (if any added later) would come from args_list below

    # Add specific arguments passed via args_list (e.g., -n, -l for predict_edits; -n for download_edits)
    # This needs to happen *after* setting script-specific defaults above
    if args_list:
        command.extend(args_list)


    print(f"\n--- Running {script_name} ---")
    print(f"Command: {' '.join(command)}")
    try:
        # Using check=True raises CalledProcessError if the script exits with non-zero status
        subprocess.run(command, check=True, text=True, cwd=BASE_DIR)
        print(f"--- {script_name} completed successfully ---")
        return True
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error: {script_name} failed with exit code {e.returncode}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred running {script_name}: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the full Wikipedia edit analysis pipeline for a single article.")
    parser.add_argument("article_title", help="The title of the Wikipedia article.")
    parser.add_argument("--num_revisions", type=int, default=100, help="Number of revisions to fetch (default: 100).")
    parser.add_argument("--num_predictions", type=int, default=10, help="Number of future edits to predict (default: 10).")
    parser.add_argument("--look_back", type=int, default=10, help="Number of recent edits for prediction trend (default: 10).")
    parser.add_argument("--file-prefix", default="", help="Prefix for intermediate and output filenames (used in batch mode).")
    # Add other parameters if needed

    args = parser.parse_args()
    file_prefix = args.file_prefix # Get the prefix for this run

    print(f"Starting analysis pipeline for article: '{args.article_title}' (Prefix: '{file_prefix}')")

    # Step 1: Download Edits
    download_args = ["-n", str(args.num_revisions), "--output_dir", DATA_DIR]
    if not run_step("download_edits.py", args.article_title, file_prefix, download_args):
        print("Pipeline aborted due to download failure.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Vectorize Edits
    if not run_step("vectorize_edits.py", args.article_title, file_prefix):
        print("Pipeline aborted due to vectorization failure.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Predict Edits (Renumbered from Step 4)
    predict_args = ["-n", str(args.num_predictions), "-l", str(args.look_back)]
    if not run_step("predict_edits.py", args.article_title, file_prefix, predict_args):
        print("Pipeline aborted due to prediction failure.", file=sys.stderr)
        sys.exit(1)

    # Step 4: Visualize Edits (Renumbered from Step 5)
    if not run_step("visualize_edits.py", args.article_title, file_prefix):
        print("Visualization step failed.", file=sys.stderr)
        sys.exit(1)

    print("\n--- Pipeline finished successfully! ---")
    print(f"Intermediate data saved in: {DATA_DIR} (using prefix '{file_prefix}')")
    # Update final message to reflect prefixed HTML output
    final_output_html = f"{file_prefix}_edit_path.html" if file_prefix else "edit_path.html"
    final_output_score = f"{file_prefix}_score.json" if file_prefix else "score.json"
    print(f"Final interactive visualization saved in: {os.path.join(OUTPUT_DIR, final_output_html)}")
    print(f"Prediction score saved in: {os.path.join(OUTPUT_DIR, final_output_score)}")


if __name__ == "__main__":
    main()
