#!/usr/bin/env python3
# data_preparation.py - Script to clean and preprocess Wikipedia talk page data from convokit (wiki-corpus)

import argparse
import re
import warnings
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os
from convokit import Corpus, download

# Suppress BeautifulSoup warning about markup resembling a locator
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

# Download required NLTK data (run once)
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # Ensure punkt_tab is downloaded
        nltk.download('stopwords', quiet=True)
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Warning: NLTK data download failed: {e}. Continuing without NLTK features.")

def clean_wikipedia_text(text, use_nltk=True):
    """
    Clean Wikipedia talk page text by removing noise and standardizing format.

    Args:
        text (str): Raw Wikipedia text entry.
        use_nltk (bool): Whether to use NLTK for tokenization and stopword removal (default: True).

    Returns:
        str: Cleaned text, or original text if cleaning fails entirely.
    """
    if not text or not isinstance(text, str):
        return ""

    # Pre-filter problematic patterns (e.g., malformed XML/HTML comments, tags, or incomplete markup)
    text = re.sub(r'<!\[.*?\]', '', text)  # Remove malformed XML/HTML comments like <![...]
    text = re.sub(r'</?\s*[^>]*\s*[^>]*>', '', text)  # Remove malformed or incomplete tags (more lenient)
    text = re.sub(r'<\s*![^>]*>', '', text)  # Remove any remaining XML declarations or comments

    cleaned_text = text  # Fallback in case BeautifulSoup fails

    try:
        # Remove HTML tags using BeautifulSoup with lxml parser (more lenient with malformed markup)
        soup = BeautifulSoup(text, 'lxml')  # Use lxml for better handling of malformed markup
        cleaned_text = soup.get_text()
    except Exception as e:
        print(f"Warning: BeautifulSoup parsing failed for text: {text[:50]}... Skipping HTML parsing. Error: {e}")
        # Use pre-filtered text as fallback
        cleaned_text = text

    # Remove wiki markup (e.g., [[...]], {{...}}, <...>)
    cleaned_text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', cleaned_text)  # Remove double brackets, keep content
    cleaned_text = re.sub(r'{{([^{}]+)}}', '', cleaned_text)          # Remove templates
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)               # Remove HTML-like tags (already handled by BeautifulSoup, but as fallback)

    # Remove URLs
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)

    # Remove excessive whitespace and normalize
    cleaned_text = ' '.join(cleaned_text.split())

    # Remove special characters and punctuation (optional—keep if punctuation is meaningful for sentiment/context)
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))

    # Convert to lowercase for consistency
    cleaned_text = cleaned_text.lower()

    # Optional: Remove stopwords and tokenize (if NLTK is available and use_nltk is True)
    if use_nltk:
        try:
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(cleaned_text)
            cleaned_text = ' '.join([word for word in tokens if word not in stop_words])
        except LookupError as e:
            print(f"Warning: NLTK resource missing or unavailable: {e}. Skipping NLTK processing.")
            use_nltk = False

    return cleaned_text

def load_convokit_corpus():
    """
    Load the wiki-corpus using convokit.

    Returns:
        list: List of text entries (utterances) from the corpus.
    """
    try:
        # Download and load the wiki-corpus
        corpus = Corpus(filename=download("wiki-corpus"))
        utterances = list(corpus.iter_utterances())
        # Extract text from utterances (assuming 'text' is the attribute for content)
        corpus_text = [utterance.text for utterance in utterances if utterance.text]
        return corpus_text
    except Exception as e:
        print(f"Error loading convokit corpus: {e}")
        return []

def clean_corpus(corpus, output_file='cleaned_wiki_corpus.txt', use_nltk=True):
    """
    Clean an entire corpus (list of strings) and save to a file.

    Args:
        corpus (list): List of raw Wikipedia text entries.
        output_file (str): Path to save the cleaned corpus (default: 'cleaned_wiki_corpus.txt').
        use_nltk (bool): Whether to use NLTK for tokenization and stopword removal (default: True).
    """
    cleaned_corpus = []
    for entry in corpus:
        cleaned_entry = clean_wikipedia_text(entry, use_nltk)
        cleaned_corpus.append(cleaned_entry)

    # Save cleaned corpus to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for cleaned_text in cleaned_corpus:
            f.write(cleaned_text + '\n')

    # Print summary and sample
    print(f"\nData Preparation Complete!")
    print(f"Total entries processed: {len(cleaned_corpus)}")
    print(f"Sample cleaned entry: {cleaned_corpus[0] if cleaned_corpus else 'No entries'}")
    print(f"Cleaned corpus saved to: {os.path.abspath(output_file)}")

def main():
    """
    Main function to run the data preparation pipeline using convokit wiki-corpus.
    """
    parser = argparse.ArgumentParser(description="Clean the convokit wiki-corpus and save it as a text file.")
    parser.add_argument("--out", default="data/cleaned_wiki_corpus.txt", help="Output path for the cleaned corpus.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Download required NLTK data (run once)
    download_nltk_data()

    # Load corpus from convokit
    corpus = load_convokit_corpus()
    
    if not corpus:
        print("No corpus loaded. Exiting.")
        return
    
    # Clean the corpus and save to file, with NLTK processing if available
    clean_corpus(corpus, output_file=args.out, use_nltk=True)

if __name__ == "__main__":
    main()