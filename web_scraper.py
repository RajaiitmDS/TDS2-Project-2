import trafilatura
import requests
import logging
from bs4 import BeautifulSoup
import pandas as pd
import re

logger = logging.getLogger(__name__)

def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Some common website to crawl information from:
    MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        raise

def scrape_wikipedia_table(url: str) -> pd.DataFrame:
    """
    Scrape tables from Wikipedia pages and return as pandas DataFrame
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse with BeautifulSoup for better table handling
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table', {'class': 'wikitable'})
        
        if not tables:
            # Fallback to any table
            tables = soup.find_all('table')
        
        if not tables:
            raise ValueError("No tables found on the page")
        
        # Use the largest table (most likely the main data table)
        largest_table = max(tables, key=lambda t: len(t.find_all('tr')))
        
        # Convert to DataFrame
        df = pd.read_html(str(largest_table))[0]
        
        # Clean up column names
        df.columns = [str(col).strip() for col in df.columns]
        
        logger.info(f"Scraped table with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to scrape Wikipedia table from {url}: {e}")
        raise

def extract_highest_grossing_films(url: str = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films") -> pd.DataFrame:
    """
    Specifically extract the highest grossing films table from Wikipedia
    """
    try:
        # Use pandas read_html for Wikipedia tables
        tables = pd.read_html(url)
        
        # Find the main table with rank information (first table is usually the main one)
        main_table = None
        
        # Look for tables with rank/position information first
        for table in tables:
            if table.shape[1] >= 4 and table.shape[0] > 25:  # Should have multiple columns and many rows
                # Check columns for rank-like data
                first_col_name = str(table.columns[0]).lower()
                if 'rank' in first_col_name or 'position' in first_col_name or table.columns[0] == 0:
                    main_table = table
                    break
                # Also check if first column contains numeric ranking data
                if pd.api.types.is_numeric_dtype(table.iloc[:, 0]):
                    first_col_values = table.iloc[:5, 0].tolist()  # Check first 5 values
                    if all(isinstance(v, (int, float)) and v <= 100 for v in first_col_values if pd.notna(v)):
                        main_table = table
                        break
        
        # Fallback: find any table with film data
        if main_table is None:
            for table in tables:
                if table.shape[1] >= 4 and table.shape[0] > 50:
                    columns_str = ' '.join([str(col).lower() for col in table.columns])
                    if any(keyword in columns_str for keyword in ['film', 'movie', 'title', 'gross', 'worldwide']):
                        main_table = table
                        break
        
        if main_table is None:
            raise ValueError("Could not find the main films table")
        
        # Clean up the dataframe
        df = main_table.copy()
        
        # Clean column names - handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level columns
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        
        # Standardize column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Try to identify key columns
        rank_cols = [col for col in df.columns if 'rank' in col.lower()]
        title_cols = [col for col in df.columns if any(word in col.lower() for word in ['title', 'film', 'movie'])]
        gross_cols = [col for col in df.columns if 'gross' in col.lower() or 'worldwide' in col.lower()]
        year_cols = [col for col in df.columns if 'year' in col.lower()]
        peak_cols = [col for col in df.columns if 'peak' in col.lower()]
        
        logger.info(f"Found columns: {df.columns.tolist()}")
        logger.info(f"Rank columns: {rank_cols}")
        logger.info(f"Title columns: {title_cols}")
        logger.info(f"Gross columns: {gross_cols}")
        logger.info(f"Year columns: {year_cols}")
        logger.info(f"Peak columns: {peak_cols}")
        
        # Clean up the data
        # Remove rows that are mostly NaN
        df = df.dropna(how='all')
        
        # Add Rank column if it doesn't exist (use index + 1)
        if not rank_cols:
            df.insert(0, 'Rank', range(1, len(df) + 1))
            rank_cols = ['Rank']
            logger.info("Added synthetic Rank column")
        
        # Add Peak column if it doesn't exist (use rank as peak for simplicity)
        if not peak_cols and rank_cols:
            df['Peak'] = df[rank_cols[0]]
            peak_cols = ['Peak']
            logger.info("Added synthetic Peak column")
        
        # Convert gross amounts to numeric where possible
        for col in gross_cols:
            if col in df.columns:
                # Clean currency symbols and convert to numeric
                df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert years to numeric
        for col in year_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert rank and peak to numeric
        for col in rank_cols + peak_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Processed films table with shape {df.shape}")
        logger.info(f"Final columns: {df.columns.tolist()}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to extract highest grossing films: {e}")
        raise
