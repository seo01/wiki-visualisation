import numpy as np
import argparse
import os
import sys

def load_vectors(input_path):
    """Loads the historical edit vectors from a .npy file."""
    try:
        vectors = np.load(input_path)
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise ValueError("Invalid format: Expected a 2D NumPy array.")
        print(f"Successfully loaded historical vectors with shape {vectors.shape} from {input_path}")
        return vectors
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"Error loading or validating vectors: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}", file=sys.stderr)
        return None

def predict_future_vectors(vectors, num_predictions=5, look_back=10):
    """
    Predicts future vectors using simple linear extrapolation based on recent trends.

    Args:
        vectors (np.ndarray): The historical sequence of vectors.
        num_predictions (int): The number of future vectors to predict.
        look_back (int): How many recent vectors to use for calculating the average difference.

    Returns:
        np.ndarray: An array containing the 'num_predictions' predicted future vectors,
                    based on the trend derived from vectors[:-num_predictions].
                    Returns None if inputs are insufficient.
    """
    total_historical = vectors.shape[0]
    if vectors is None or total_historical < 2:
        print("Error: Need at least two historical vectors overall.", file=sys.stderr)
        return None

    # Determine the range of vectors to use for trend calculation
    train_end_index = total_historical - num_predictions
    if train_end_index < 2:
         print(f"Error: Not enough data ({train_end_index}) before the prediction window to calculate a trend.", file=sys.stderr)
         return None

    train_vectors = vectors[:train_end_index]
    start_prediction_vector = train_vectors[-1] # Predict starting from the last vector used for training trend

    # Adjust look_back if it exceeds available training data
    if train_vectors.shape[0] <= look_back:
        print(f"Warning: Not enough training data ({train_vectors.shape[0]}) for the requested look_back ({look_back}). Using all available training data for trend.", file=sys.stderr)
        look_back = train_vectors.shape[0] - 1 # Need at least one difference for trend calculation

    if look_back < 1:
        print("Error: Adjusted look_back is less than 1. Cannot calculate trend.", file=sys.stderr)
        return None

    print(f"Predicting {num_predictions} vectors, using trend from vectors 0 to {train_end_index-1} (look_back={look_back})...")

    # Calculate differences within the look_back window *within the training data*
    trend_vectors = train_vectors[-look_back-1:] # Get last look_back+1 vectors *from training set*
    differences = np.diff(trend_vectors, axis=0)

    # Calculate the average difference (the trend)
    avg_diff = np.mean(differences, axis=0)

    # Extrapolate future vectors starting from the end of the training data
    predicted_vectors = []
    current_vector = start_prediction_vector
    for i in range(num_predictions):
        next_vector = current_vector + avg_diff
        predicted_vectors.append(next_vector)
        current_vector = next_vector # Use the predicted vector as the base for the next prediction

    print(f"Prediction complete. Generated {len(predicted_vectors)} predicted vectors.")
    return np.array(predicted_vectors)

def save_vectors(vectors, output_path):
    """Saves the predicted vector matrix to a .npy file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, vectors)
        print(f"Predicted vectors saved successfully to {output_path}")
    except IOError as e:
        print(f"Error saving predicted vectors to {output_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Predict future Wikipedia edit vectors using simple extrapolation.")
    parser.add_argument("-n", "--num_predictions", type=int, default=10, help="Number of future edits to predict (default: 10).")
    parser.add_argument("-l", "--look_back", type=int, default=10, help="Number of recent edits to base the trend on (default: 10).")
    parser.add_argument("-i", "--input_dir", default="data", help="Directory containing the edit vectors NPY file (default: data).")
    parser.add_argument("-o", "--output_dir", default="data", help="Directory to save the predicted vectors NPY file (default: data).")
    parser.add_argument("--file-prefix", default="", help="Prefix for input/output filenames.")
    args = parser.parse_args()

    input_filename = f"{args.file_prefix}_edit_vectors.npy" if args.file_prefix else "edit_vectors.npy"
    output_filename = f"{args.file_prefix}_predicted_vectors.npy" if args.file_prefix else "predicted_vectors.npy"
    input_path = os.path.join(args.input_dir, input_filename)
    output_path = os.path.join(args.output_dir, output_filename)

    historical_vectors = load_vectors(input_path)

    if historical_vectors is not None:
        predicted_vectors = predict_future_vectors(historical_vectors, args.num_predictions, args.look_back)
        if predicted_vectors is not None:
            save_vectors(predicted_vectors, output_path)
        else:
            print("Failed to predict future vectors.")
            sys.exit(1)
    else:
        print("Failed to load historical vectors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
