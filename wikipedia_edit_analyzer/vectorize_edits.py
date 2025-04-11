import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import os
import sys

def load_revisions(input_path):
    """Loads revision texts from a JSON file."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            revisions = json.load(f)
        if not isinstance(revisions, list) or not all(isinstance(item, str) for item in revisions):
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

def vectorize_text(texts):
    """Converts a list of text documents into a TF-IDF matrix."""
    if not texts:
        print("Warning: No texts provided for vectorization.", file=sys.stderr)
        return None
    try:
        print(f"Vectorizing {len(texts)} text documents using TF-IDF...")
        # Using default settings for TfidfVectorizer. Can be tuned (e.g., max_features, stop_words)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        print(f"Vectorization complete. Matrix shape: {tfidf_matrix.shape}")
        # Convert sparse matrix to dense numpy array for easier handling later
        return tfidf_matrix.toarray()
    except Exception as e:
        print(f"An error occurred during TF-IDF vectorization: {e}", file=sys.stderr)
        return None

def save_vectors(vectors, output_path):
    """Saves the vector matrix to a .npy file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, vectors)
        print(f"Vectors saved successfully to {output_path}")
    except IOError as e:
        print(f"Error saving vectors to {output_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Vectorize Wikipedia article revisions using TF-IDF.")
    parser.add_argument("-i", "--input_dir", default="data", help="Directory containing the raw edits JSON file (default: data).")
    parser.add_argument("-o", "--output_dir", default="data", help="Directory to save the output vectors NPY file (default: data).")
    parser.add_argument("--file-prefix", default="", help="Prefix for input/output filenames.")
    args = parser.parse_args()

    input_filename = f"{args.file_prefix}_raw_edits.json" if args.file_prefix else "raw_edits.json"
    output_filename = f"{args.file_prefix}_edit_vectors.npy" if args.file_prefix else "edit_vectors.npy"
    input_path = os.path.join(args.input_dir, input_filename)
    output_path = os.path.join(args.output_dir, output_filename)

    revisions = load_revisions(input_path)

    if revisions:
        vectors = vectorize_text(revisions)
        if vectors is not None:
            save_vectors(vectors, output_path)
        else:
            print("Failed to vectorize revisions.")
            sys.exit(1)
    else:
        print("Failed to load revisions.")
        sys.exit(1)

if __name__ == "__main__":
    main()
