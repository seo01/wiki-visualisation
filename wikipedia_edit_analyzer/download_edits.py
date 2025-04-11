import requests
import json
import argparse
import os
import sys
import time

# MediaWiki API endpoint
API_URL = "https://en.wikipedia.org/w/api.php"

# Define a user agent (replace with more specific info if needed)
HEADERS = {
    'User-Agent': 'WikipediaEditAnalyzer/0.1 (https://example.com/contact; cooltool@example.com) requests'
}

# Rate limiting: Wait this many seconds between API calls
API_DELAY = 0.5 # Adjust as needed, be respectful of API limits

def fetch_revision_ids(session, article_title, num_revisions):
    """Fetches the IDs of the last 'num_revisions' for an article."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": article_title,
        "rvprop": "ids",
        "rvlimit": num_revisions,
        "rvdir": "older", # Get newest first
        "redirects": 1 # Follow redirects
    }
    try:
        time.sleep(API_DELAY) # Rate limiting
        response = session.get(API_URL, params=params, headers=HEADERS)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Navigate the complex JSON structure
        query_pages = data.get("query", {}).get("pages")
        if not query_pages:
            raise ValueError("API response missing 'pages' data.")

        # Find the page data (usually only one page unless title resolves weirdly)
        page_id = list(query_pages.keys())[0]
        if page_id == "-1": # Page not found indicator
             print(f"Error: Page '{article_title}' not found via API.", file=sys.stderr)
             return None

        page_data = query_pages[page_id]
        revisions = page_data.get("revisions")
        if not revisions:
            print(f"Warning: No revisions found for '{article_title}'.", file=sys.stderr)
            return [] # Return empty list if no revisions

        revision_ids = [rev["revid"] for rev in revisions]
        print(f"Found {len(revision_ids)} revision IDs.")
        return revision_ids

    except requests.exceptions.RequestException as e:
        print(f"API request failed when fetching revision IDs: {e}", file=sys.stderr)
        return None
    except (KeyError, IndexError, ValueError) as e:
         print(f"Error parsing API response for revision IDs: {e}", file=sys.stderr)
         print("Response data:", data) # Log the problematic data
         return None
    except Exception as e:
         print(f"An unexpected error occurred fetching revision IDs: {e}", file=sys.stderr)
         return None


def fetch_revision_content(session, revision_ids, batch_size=50):
    """Fetches the content for a list of revision IDs, handling batching."""
    all_revision_content = {} # Use dict to store content by revid initially

    if not revision_ids:
        return []

    x = 0
    print(f"Fetching content for {len(revision_ids)} revisions...")
    for i in range(0, len(revision_ids), batch_size):
        batch_ids = revision_ids[i:i+batch_size]
        print(f"Fetching content batch {i//batch_size + 1} ({len(batch_ids)} IDs)...")

        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "revids": "|".join(map(str, batch_ids)),
            "rvprop": "content",
            "rvslots": "main" # Specify the main content slot
        }
        try:
            time.sleep(API_DELAY) # Rate limiting
            response = session.get(API_URL, params=params, headers=HEADERS)
            response.raise_for_status()
            data = response.json()

            query_pages = data.get("query", {}).get("pages")
            if not query_pages:
                raise ValueError("API response missing 'pages' data.")

            for page_id, page_data in query_pages.items():
                revisions_data = page_data.get("revisions")
                if revisions_data:
                    for rev_data in revisions_data:
                        rev_id = revision_ids[x] #rev_data.get("revid")
                        x +=1
                        content = None
                        # Try extracting from slots first (current standard)
                        slots = rev_data.get("slots")
                        if slots:
                            main_slot = slots.get("main")
                            if main_slot:
                                content = main_slot.get("*")
                        
                        # Fallback: Check for content directly under the revision if not found in slots
                        if content is None:
                            content = rev_data.get("*")

                        # Store if rev_id and content are valid
                        if rev_id and content is not None:
                            all_revision_content[rev_id] = content
                        else:
                            # Improved warning message
                            print(f"Warning: Missing content for revision ID {rev_id if rev_id else 'Unknown'} in batch {i//batch_size + 1}.", file=sys.stderr)
                            # Optionally log more details about rev_data here if debugging further
                            # print(f"Debug: rev_data for missing content: {rev_data}")


        except requests.exceptions.RequestException as e:
            print(f"API request failed fetching content batch {i//batch_size + 1}: {e}", file=sys.stderr)
            # Continue to next batch if possible, or decide to fail entirely
        except (KeyError, ValueError) as e:
             print(f"Error parsing API response for content batch {i//batch_size + 1}: {e}", file=sys.stderr)
             print("Response data:", data)
        except Exception as e:
             print(f"An unexpected error occurred fetching content batch {i//batch_size + 1}: {e}", file=sys.stderr)

    # Return content in the original order of revision IDs
    ordered_content = [all_revision_content.get(rev_id) for rev_id in revision_ids]
    # Filter out None values if some revisions failed
    final_content = [content for content in ordered_content if content is not None]
    print(f"Successfully fetched content for {len(final_content)} out of {len(revision_ids)} revisions.")
    return final_content


def save_revisions(revisions, output_path):
    """Saves the list of revision content to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(revisions, f, ensure_ascii=False, indent=4)
        print(f"Revisions saved successfully to {output_path}")
    except IOError as e:
        print(f"Error saving revisions to {output_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia article revisions using MediaWiki API.")
    parser.add_argument("article_title", help="The title of the Wikipedia article.")
    parser.add_argument("-n", "--num_revisions", type=int, default=100, help="Number of revisions to fetch (default: 100).")
    parser.add_argument("-o", "--output_dir", default="data", help="Directory to save the output JSON file (default: data).")
    parser.add_argument("--file-prefix", default="", help="Prefix for output filenames (e.g., derived from article title).")
    args = parser.parse_args()

    output_filename = f"{args.file_prefix}_raw_edits.json" if args.file_prefix else "raw_edits.json"
    output_path = os.path.join(args.output_dir, output_filename)

    session = requests.Session() # Use a session for potential connection reuse

    print(f"Fetching revision IDs for '{args.article_title}'...")
    revision_ids = fetch_revision_ids(session, args.article_title, args.num_revisions)

    revisions_content = []
    if revision_ids is not None: # Proceed even if revision_ids is empty
        if revision_ids:
            revisions_content = fetch_revision_content(session, revision_ids)
        else:
             print("No revision IDs found, cannot fetch content.")
    else:
         print("Failed to fetch revision IDs.")
         sys.exit(1) # Exit if ID fetching failed critically

    if revisions_content: # Only save if we got some content
        save_revisions(revisions_content, output_path)
    else:
        print("No revision content was successfully fetched or saved.")
        # Decide if this is an error or just an empty result case
        if revision_ids is None: # If ID fetch failed, it's an error
             sys.exit(1)
        # If IDs were fetched but content failed/was empty, maybe exit 0? For now, exit 1.
        sys.exit(1)


if __name__ == "__main__":
    main()
