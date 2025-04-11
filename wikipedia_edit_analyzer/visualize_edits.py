import numpy as np
import umap
import plotly.graph_objects as go
import argparse
import os
import sys
import json # Needed to load diffs and save score

def load_vectors(file_path):
    """Loads vectors from a .npy file."""
    try:
        vectors = np.load(file_path)
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
             raise ValueError("Invalid format: Expected a 2D NumPy array.")
        print(f"Successfully loaded vectors with shape {vectors.shape} from {file_path}")
        return vectors
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"Error loading or validating vectors from {file_path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path}: {e}", file=sys.stderr)
        return None

def load_diffs(file_path):
    """Loads diff texts from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            diffs = json.load(f)
        if not isinstance(diffs, list) or (diffs and not all(isinstance(item, str) for item in diffs)):
            raise ValueError("Invalid format: Expected a list of strings in the JSON file.")
        print(f"Successfully loaded {len(diffs)} diffs from {file_path}")
        return diffs
    except FileNotFoundError:
        print(f"Error: Diffs file not found at {file_path}. Hover text will be limited.", file=sys.stderr)
        return None # Proceed without diffs if not found
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Hover text will be limited.", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"Error: {e}. Hover text will be limited.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading diffs: {e}. Hover text will be limited.", file=sys.stderr)
        return None

def format_hover_text(diffs, num_total_historical, num_predicted_tail):
    """Formats diffs for Plotly hover text for the fork visualization."""
    hover_texts = []
    
    # Historical points (base + actual tail)
    for i in range(num_total_historical):
        if diffs and i < len(diffs):
            text = diffs[i].replace('\n', '<br>')
            # Add truncation if needed
            # max_len = 500 ... text = text[:max_len] + "... (truncated)"
            hover_texts.append(text)
        else:
             hover_texts.append(f"Revision {i}" if i > 0 else "Initial Revision")

    # Predicted points
    for i in range(num_predicted_tail):
        hover_texts.append(f"Predicted Edit {i+1} (from step {num_total_historical - num_predicted_tail})")
        
    # Expected length: num_total_historical + num_predicted_tail
    return hover_texts


def reduce_dimensions(historical_vectors, predicted_vectors, n_components=2, **umap_kwargs):
    """
    Reduces the dimensionality of combined historical and predicted vectors using UMAP.
    Ensures predicted_vectors is handled correctly, even if None.
    """
    if historical_vectors is None:
        print("Error: Historical vectors are required for dimensionality reduction.", file=sys.stderr)
        return None, 0 # Return embedding and num_historical

    num_historical = historical_vectors.shape[0]
    
    # Combine historical and predicted vectors for joint reduction
    if predicted_vectors is not None and predicted_vectors.shape[0] > 0:
        if historical_vectors.shape[1] != predicted_vectors.shape[1]:
            print("Error: Historical and predicted vectors have different feature dimensions.", file=sys.stderr)
            return None, num_historical
        combined_vectors = np.vstack((historical_vectors, predicted_vectors))
        print(f"Combined historical ({num_historical}) and predicted ({predicted_vectors.shape[0]}) vectors for UMAP.")
    else:
        combined_vectors = historical_vectors
        print("Only historical vectors provided for UMAP.")

    try:
        print(f"Reducing dimensions to {n_components} using UMAP...")
        reducer = umap.UMAP(n_components=n_components, random_state=42, **umap_kwargs) # Added random_state
        embedding = reducer.fit_transform(combined_vectors)
        print(f"UMAP reduction complete. Output shape: {embedding.shape}")
        return embedding, num_historical # Return embedding and count
    except Exception as e:
        print(f"An error occurred during UMAP reduction: {e}", file=sys.stderr)
        return None, num_historical


def plot_interactive_fork_path(fig, embedding, num_historical, num_predicted, hover_texts, title="Wikipedia Article Edit Path (UMAP)"):
    """
    Populates a Plotly figure object with the 2D edit path fork and calculates score.

    Args:
        fig (go.Figure): The Plotly figure object to populate.
        embedding (np.ndarray): The 2D vector representations for ALL points (hist + pred).
        num_historical (int): The total number of historical points.
        num_predicted (int): The number of points in the predicted tail.
        hover_texts (list): List of strings for hover text (length should match embedding rows).
        title (str): The title for the plot.

    Returns:
        float or None: The calculated Euclidean distance score, or None if calculation failed.
    """
    if embedding is None or embedding.shape[1] != 2:
        print("Error: Invalid embedding provided for plotting.", file=sys.stderr)
        return None
    if num_predicted <= 0 or num_historical <= num_predicted:
        print("Error: Insufficient historical data or zero predicted points for fork visualization.", file=sys.stderr)
        return None

    fork_point_index = num_historical - num_predicted -1
    if fork_point_index < 0:
         print("Error: Calculation error, fork point index is negative.", file=sys.stderr)
         return None

    num_total_points = embedding.shape[0]
    expected_hover_len = num_historical + num_predicted
    if hover_texts is None or len(hover_texts) != expected_hover_len:
        print(f"Warning: Hover text length mismatch (expected {expected_hover_len}, got {len(hover_texts) if hover_texts else 0}). Using default labels.", file=sys.stderr)
        hover_texts = format_hover_text(None, num_historical, num_predicted)

    # 1. Base Historical Path
    base_indices = np.arange(0, fork_point_index + 1)
    base_hover = hover_texts[:fork_point_index + 1]
    fig.add_trace(go.Scattergl(
        x=embedding[base_indices, 0], y=embedding[base_indices, 1], mode='lines+markers',
        marker=dict(size=6, symbol='circle'), line=dict(width=1.5), name='Historical Path (Base)',
        hoverinfo='text', hovertext=base_hover, hovertemplate='<b>Diff to this Revision:</b><br>%{hovertext}<extra></extra>'
    ))

    # 2. Actual Tail Path
    actual_tail_indices = np.arange(fork_point_index, num_historical)
    actual_tail_hover = hover_texts[fork_point_index:num_historical]
    fig.add_trace(go.Scattergl(
        x=embedding[actual_tail_indices, 0], y=embedding[actual_tail_indices, 1], mode='lines+markers',
        marker=dict(size=8, symbol='circle'), line=dict(width=2.5), name='Actual Edit Path (Tail)',
        hoverinfo='text', hovertext=actual_tail_hover, hovertemplate='<b>Diff to this Revision:</b><br>%{hovertext}<extra></extra>'
    ))

    # 3. Predicted Tail Path
    predicted_coords_indices = np.arange(num_historical, num_total_points)
    predicted_plot_x = np.concatenate(([embedding[fork_point_index, 0]], embedding[predicted_coords_indices, 0]))
    predicted_plot_y = np.concatenate(([embedding[fork_point_index, 1]], embedding[predicted_coords_indices, 1]))
    predicted_plot_hover = [hover_texts[fork_point_index]] + hover_texts[num_historical:]
    fig.add_trace(go.Scattergl(
        x=predicted_plot_x, y=predicted_plot_y, mode='lines+markers',
        marker=dict(size=9, symbol='cross'), line=dict(width=2.5, dash='dash'), name='Predicted Edit Path (Tail)',
        hoverinfo='text', hovertext=predicted_plot_hover, hovertemplate='<b>%{hovertext}</b><extra></extra>'
    ))

    # Key point markers
    fig.add_trace(go.Scattergl(x=[embedding[0, 0]], y=[embedding[0, 1]], mode='markers', marker=dict(color='black', size=12, symbol='star'), name='Start', hoverinfo='skip'))
    fig.add_trace(go.Scattergl(x=[embedding[fork_point_index, 0]], y=[embedding[fork_point_index, 1]], mode='markers', marker=dict(color='red', size=14, symbol='diamond'), name='Prediction Start / Fork Point', hoverinfo='skip'))
    fig.add_trace(go.Scattergl(x=[embedding[num_historical-1, 0]], y=[embedding[num_historical-1, 1]], mode='markers', marker=dict(size=12, symbol='circle'), name='End of Actual', hoverinfo='skip'))
    fig.add_trace(go.Scattergl(x=[embedding[-1, 0]], y=[embedding[-1, 1]], mode='markers', marker=dict(size=12, symbol='cross'), name='End of Predicted', hoverinfo='skip'))

    # Layout update
    fig.update_layout(
        title=title, xaxis_title="UMAP Dimension 1", yaxis_title="UMAP Dimension 2", hovermode='closest',
        legend_title_text='Path Segment', template='seaborn', margin=dict(l=40, r=40, t=80, b=40)
    )

    # Calculate score and add annotation
    score = None
    if num_predicted > 0 and num_historical > 0:
        final_actual_coord = embedding[num_historical - 1]
        final_pred_coord = embedding[-1]
        distance = np.linalg.norm(final_actual_coord - final_pred_coord)
        score = distance # Store the numpy float
        fig.add_annotation(
            x=0.98, y=0.01, xref="paper", yref="paper", text=f"Prediction Error (2D Distance): {distance:.3f}",
            showarrow=False, font=dict(size=10, color="black"), bgcolor="rgba(255, 255, 255, 0.7)", bordercolor="black", borderwidth=1
        )
        
    return score


def save_plot_and_score(fig, score, html_output_path, score_output_path):
    """Saves the Plotly figure as HTML and the score as JSON."""
    # Save HTML
    try:
        os.makedirs(os.path.dirname(html_output_path), exist_ok=True)
        fig.write_html(html_output_path, include_plotlyjs='cdn')
        print(f"Interactive plot saved successfully to {html_output_path}")
    except IOError as e:
        print(f"Error saving plot to {html_output_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during plot saving: {e}", file=sys.stderr)

    # Save Score
    if score is not None:
        try:
            os.makedirs(os.path.dirname(score_output_path), exist_ok=True)
            # Convert potential NumPy float to standard Python float for JSON compatibility
            save_score = float(score) 
            with open(score_output_path, 'w', encoding='utf-8') as f:
                 # Save score in a simple JSON structure
                 json.dump({"prediction_error_distance_2d": save_score}, f, indent=2)
            print(f"Score saved successfully to {score_output_path}")
        except IOError as e:
             print(f"Error saving score to {score_output_path}: {e}", file=sys.stderr)
        except Exception as e:
             print(f"An unexpected error occurred during score saving: {e}", file=sys.stderr)
    else:
        print("No score calculated or provided to save.")


def main():
    parser = argparse.ArgumentParser(description="Visualize Wikipedia edit path interactively with prediction fork.")
    parser.add_argument("-i_hist", "--input_hist_dir", default="data", help="Directory containing historical vectors NPY file (default: data).")
    parser.add_argument("-i_pred", "--input_pred_dir", default="data", help="Directory containing predicted vectors NPY file (default: data).")
    parser.add_argument("-i_diffs", "--input_diffs_dir", default="data", help="Directory containing diffs JSON file (default: data).")
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to save the output plot HTML and score JSON files (default: output).")
    parser.add_argument("-t", "--title", default="Wikipedia Article Edit Path", help="Title for the plot.")
    parser.add_argument("--file-prefix", default="", help="Prefix for input/output filenames.")
    args = parser.parse_args()

    hist_filename = f"{args.file_prefix}_edit_vectors.npy" if args.file_prefix else "edit_vectors.npy"
    pred_filename = f"{args.file_prefix}_predicted_vectors.npy" if args.file_prefix else "predicted_vectors.npy"
    diffs_filename = f"{args.file_prefix}_edit_diffs.json" if args.file_prefix else "edit_diffs.json"
    output_html_filename = f"{args.file_prefix}_edit_path.html" if args.file_prefix else "edit_path.html"
    output_score_filename = f"{args.file_prefix}_score.json" if args.file_prefix else "score.json"

    hist_path = os.path.join(args.input_hist_dir, hist_filename)
    pred_path = os.path.join(args.input_pred_dir, pred_filename)
    diffs_path = os.path.join(args.input_diffs_dir, diffs_filename)
    output_html_path = os.path.join(args.output_dir, output_html_filename)
    output_score_path = os.path.join(args.output_dir, output_score_filename) # Path for score file

    historical_vectors = load_vectors(hist_path)
    predicted_vectors = load_vectors(pred_path)
    diffs = load_diffs(diffs_path)

    if historical_vectors is None:
        print("Cannot proceed without historical vectors.", file=sys.stderr)
        sys.exit(1)
        
    num_predicted_points = predicted_vectors.shape[0] if predicted_vectors is not None else 0

    # Reduce dimensions together
    embedding, num_historical_points = reduce_dimensions(historical_vectors, predicted_vectors)

    if embedding is not None:
        # Prepare hover texts for all points (N historical + M predicted)
        hover_texts = format_hover_text(diffs, num_historical_points, num_predicted_points)

        plot_title = f"{args.title} (UMAP - Prediction Fork)"
        
        # Generate the figure object and populate it
        fig = go.Figure()
        score = plot_interactive_fork_path(fig, embedding, num_historical_points, num_predicted_points, hover_texts, title=plot_title)

        # Save the plot and the score
        save_plot_and_score(fig, score, output_html_path, output_score_path)

    else:
        print("Failed to generate 2D embedding.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
