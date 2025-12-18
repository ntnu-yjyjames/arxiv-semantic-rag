
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

def fetch_titles(limit: int = 100):
    """
    Fetches a sample of paper titles from the backend to populate UI selectors.

    This function is designed to be "fail-safe":
    - If the backend is offline or errors out, it catches the exception.
    - Displays a user-friendly `st.error` message on the frontend.
    - Returns an empty list `[]` to prevent the UI widget from crashing.

    Args:
        limit (int): Maximum number of titles to fetch.

    Returns:
        List[str]: A list of titles retrieved from `/titles`
                   (or an empty list if the fetch fails).
    """
    try:
        resp = requests.get(f"{BACKEND_URL}/titles", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch titles from backend: {e}")
        return []

