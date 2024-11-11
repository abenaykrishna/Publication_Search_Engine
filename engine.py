import requests
import time
import tqdm
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download necessary NLTK resources
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')
Authors = []
Publication_year = []
Title = []
Publication_page_link = []
Authors_profile_page_link = []
# Function to fetch publications from the specified URL page
def fetch_publications(page_number):
    URL = f'https://pureportal.coventry.ac.uk/en/organisations/centre-global-learning/publications/?page={page_number}'
    try:
        page = requests.get(URL)
        page.raise_for_status()  # Raise an error for bad responses (4XX or 5XX)
        html_code = page.text
        
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_code, 'html.parser')
        return soup  # Return the parsed HTML for further processing
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
soup = fetch_publications(1)
for x in soup.find_all('div', attrs={'class': 'result-container'}):
    authors = x.find_all('a', attrs={'class': 'person'})
    Authors.append([author.text for author in authors] if authors else np.nan)
    # Extract Publication Year
    publication_year = x.find('span', attrs={'class': 'date'})
    Publication_year.append(publication_year.text if publication_year else np.nan)  # Changed np.NaN to np.nan
    # Extract Title
    title = x.find('h3', attrs={'class': 'title'})
    Title.append(title.text if title else np.nan)  # Changed np.NaN to np.nan

    # Extract Publication Page Link
    publication_page_link = x.find('a', attrs={'class': 'link'})
    Publication_page_link.append(publication_page_link['href'] if publication_page_link else np.nan)  # Changed np.NaN to np.nan

    # Extract Authors Profile Page Links
    authors_profile_page_link = x.find_all('a', attrs={'class': 'link person'})
    Authors_profile_page_link.append([link['href'] for link in authors_profile_page_link] if authors_profile_page_link else np.nan) 
print(Authors)
print(Publication_year)
print(Title)
print(Publication_page_link)
print(Authors_profile_page_link)
df1 = pd.DataFrame({
    'Authors': Authors,
    'Publication_year': Publication_year,
    'Title': Title,
    'Publication_page_link': Publication_page_link,
    'Authors_profile_page_link': Authors_profile_page_link,
})
print(df1.head())
# 1. Process the Authors column
df1['Authors'] = df1['Authors'].apply(lambda names: ', '.join(names) if isinstance(names, list) else names)
df1['Authors'] = df1['Authors'].str.replace(r'[,.]', '', regex=True)

# 2. Format the Authors_profile_page_link as clickable HTML links
def format_links(links_list):
    return ', '.join([f'<a href="{link}">{link}</a>' for link in links_list])

df1['Authors_profile_page_link'] = df1['Authors_profile_page_link'].apply(
    lambda links: format_links(links) if isinstance(links, list) else links
)

# 3. Clean up <a> tags if needed
def remove_a_tags(text):
    if isinstance(text, str):
        return re.sub(r'<a[^>]*>(.*?)<\/a>', r'\1', text)
    return text

df1['Authors_profile_page_link'] = df1['Authors_profile_page_link'].apply(remove_a_tags)

# 4. Handle single quotes in URLs
def remove_single_quotes(link_str):
    if isinstance(link_str, str):
        return re.sub(r"'", '', link_str)
    return link_str

df1['Authors_profile_page_link'] = df1['Authors_profile_page_link'].apply(remove_single_quotes)

# 5. Drop duplicates and handle NaN values
df1.drop_duplicates(keep='first', inplace=True)
df1.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df1.dropna(inplace=True)

# 6. Validate URLs in both Publication_page_link and Authors_profile_page_link columns
link_pattern = r'^https?://[\w\-]+(\.[\w\-]+)+[/#?]?.*$'

# Check if columns contain valid URLs and store them in new columns
df1['Publication_page_link_valid'] = df1['Publication_page_link'].str.match(link_pattern).fillna(False)
df1['Authors_profile_page_link_valid'] = df1['Authors_profile_page_link'].str.match(link_pattern).fillna(False)

# Create new columns for valid links
df1['Valid_Publication_page_link'] = np.where(df1['Publication_page_link_valid'], df1['Publication_page_link'], None)
df1['Valid_Authors_profile_page_link'] = np.where(df1['Authors_profile_page_link_valid'], df1['Authors_profile_page_link'], None)

# Filtering only valid URLs (if desired)
valid_df = df1[df1['Publication_page_link_valid'] & df1['Authors_profile_page_link_valid']]

# Final cleaned dataframe with validation columns retained
df1.to_csv("publication.csv")

df2 = pd.read_csv('publication.csv')
# Initialize the Flask app
app = Flask(__name__)

# Step 1: Create the index
index = defaultdict(list)


# Step 2: Iterate through the DataFrame and populate the index
for idx, row in df2.iterrows():
    doc_id = idx  # The index of the row in the DataFrame serves as the document ID
    authors = row['Authors']
    title = row['Title']
    
    # Check if authors and title are not NaN (not missing values)
    if pd.notna(authors):
        authors = authors.split()  # Split the authors into individual words
    else:
        authors = []
        
    if pd.notna(title):
        title = title.split()  # Split the title into individual words
    else:
        title = []
    
    # Update the index with each word and its corresponding document ID
    for word in authors + title:
        index[word].append(doc_id)

# Now, the 'index' dictionary contains words as keys and a list of corresponding document IDs as values.


# Function to perform a search and return the matching documents
def search(query):
    query_terms = query.lower().split()  # Preprocess the query in the same way as the documents
    matching_doc_ids = set()  # Use a set to store the matching document IDs to avoid duplicates
    
    # Iterate through each query term and find matching document IDs from the index
    for term in query_terms:
        doc_ids = index.get(term, [])  # Get the list of document IDs for the query term
        matching_doc_ids.update(doc_ids)  # Add the document IDs to the set of matching IDs
    
    # Retrieve the matching documents from the DataFrame
    matching_documents = df2.loc[list(matching_doc_ids)]
    
    return matching_documents




# Function to calculate TF-IDF scores and rank the results
def search_and_rank(query, df2):
    # Replace NaN values in 'Authors' and 'Title' columns with empty strings
    df2['Authors'] = df2['Authors'].fillna('')
    df2['Title'] = df2['Title'].fillna('')

    # Combine the 'Authors' and 'Title' columns into a single text column for TF-IDF
    df2['Text'] = df2['Authors'] + ' ' + df2['Title']

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the TF-IDF vectorizer on the combined text
    tfidf_matrix = vectorizer.fit_transform(df2['Text'])

    # Convert the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate the cosine similarity between the query vector and the document vectors
    cosine_similarities = (tfidf_matrix * query_vector.T).toarray().flatten()

    # Add a new column 'Relevance' to the DataFrame with the cosine similarities
    df2['Relevance'] = cosine_similarities

    # Sort the DataFrame by the 'Relevance' column in descending order
    ranked_df = df2.sort_values(by='Relevance', ascending=False)

    # Drop the 'Text' and 'Relevance' columns to clean up the DataFrame
    ranked_df.drop(columns=['Text', 'Relevance'], inplace=True)

    return ranked_df







@app.route('/', methods=['GET', 'POST'])
def search_page():
    if request.method == 'POST':
        query = request.form['query']
        matching_results = search(query)
        ranked_results = search_and_rank(query, matching_results)
        
        # Convert the DataFrame to a list of dictionaries for custom formatting
        formatted_results = ranked_results.to_dict(orient='records')
        
        return render_template('search_results.html', results=formatted_results)
    return render_template('search_page.html')


if __name__ == '__main__':
    app.run(debug=True)