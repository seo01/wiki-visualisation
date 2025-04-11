import json
import difflib
import argparse
import os
import sys

def load_revisions(input_path):
    """Loads revision texts from a JSON file."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            revisions = json.load(f)
        if not isinstance(revisions, list) or (revisions and not all(isinstance(item, str) for item in revisions)):
            # Allow empty list, but if not empty, must contain strings
            raise ValueError("Invalid format: Expected a list of strings in the JSON file.")
        print(f"Successfully loaded {len(revisions)} revisions from {input_path}")
        return revisions
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}", file=sys.stderr)
        return None

def calculate_diffs(revisions):
    """Calculates unified diffs between consecutive revisions."""
    if not revisions or len(revisions) < 2:
        print("Warning: Need at least two revisions to calculate diffs.", file=sys.stderr)
        # Return a list with a placeholder for the first revision if only one exists?
        # Or just return empty if less than 2? Let's return based on input.
        return ["Initial Revision"] if len(revisions) == 1 else []

    print(f"Calculating diffs for {len(revisions)} revisions...")
    diffs = ["Initial Revision"] # First revision has no preceding diff

    for i in range(len(revisions) - 1):
        rev_from = revisions[i].splitlines()
        rev_to = revisions[i+1].splitlines()
        
        # Generate unified diff
        # fromfile/tofile args are just labels used in the diff output
        diff_generator = difflib.unified_diff(rev_from, rev_to,
                                              fromfile=f'Revision {i}',
                                              tofile=f'Revision {i+1}',
                                              lineterm='')
        diff_text = '\n'.join(diff_generator)
        
        if not diff_text:
            # Handle case where revisions are identical
             diff_text = "No changes from previous revision."

        diffs.append(diff_text)
        
        if (i + 1) % 10 == 0:
             print(f"Processed diff up to revision {i+1}")


    print(f"Successfully calculated {len(diffs) -1} diffs.")
    return diffs

def save_diffs(diffs, output_path):
    """Saves the list of diff strings to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(diffs, f, ensure_ascii=False, indent=2) # Use less indent for potentially large diffs
        print(f"Diffs saved successfully to {output_path}")
    except IOError as e:
        print(f"Error saving diffs to {output_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during saving diffs: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Calculate diffs between consecutive Wikipedia revisions.")
    parser.add_argument("-i", "--input_dir", default="data", help="Directory containing the raw edits JSON file (default: data).")
    parser.add_argument("-o", "--output_dir", default="data", help="Directory to save the output diffs JSON file (default: data).")
    parser.add_argument("--file-prefix", default="", help="Prefix for input/output filenames.")
    args = parser.parse_args()

    input_filename = f"{args.file_prefix}_raw_edits.json" if args.file_prefix else "raw_edits.json"
    output_filename = f"{args.file_prefix}_edit_diffs.json" if args.file_prefix else "edit_diffs.json"
    input_path = os.path.join(args.input_dir, input_filename)
    output_path = os.path.join(args.output_dir, output_filename)

    revisions = load_revisions(input_path)

    if revisions is not None: # Proceed even if revisions list is empty or has one item
        diffs = calculate_diffs(revisions)
        # calculate_diffs returns empty list or ["Initial Revision"] if input is insufficient,
        # but saving an empty/single-item list is fine.
        save_diffs(diffs, output_path)
    else:
        print("Failed to load revisions, cannot calculate diffs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
