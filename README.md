# Sentiment Analyser for Yelp Reviews

This project is a sentiment analysis tool that processes textual reviews scraped from Yelp. It utilizes a pretrained BERT model along with NLTK for natural language processing to classify the sentiment of each review on a scale of 1 to 5. The project is implemented in Python and leverages several powerful libraries to handle data scraping, preprocessing, and deep learning-based inference.

## Features

- **Web Scraping:** Extracts review texts from Yelp using BeautifulSoup and requests.
- **Data Handling:** Uses pandas and numpy for efficient data manipulation.
- **NLP Model:** Loads a pretrained BERT model (via HuggingFace Transformers) for sentiment analysis, enhanced with NLTK for text preprocessing.
- **Deep Learning:** Utilizes PyTorch as the backend for model inference.
- **Sentiment Output:** Provides sentiment scores for each review, ranging from 1 (very negative) to 5 (very positive).

## Sentiment Output

After processing the reviews, the model predicts a sentiment rating for each review. The output is a single digit, ranging from 1 to 5, which represents the sentiment as follows:

- **1:** Very Negative (Bad)
- **2:** Negative
- **3:** Neutral
- **4:** Positive
- **5:** Very Positive (Great)

This rating system makes it easy to interpret the sentiment of each Yelp review at a glance, helping you quickly identify negative, neutral, or highly positive feedback.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kamli01/sentiment-analyser.git
   cd sentiment-analyser
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy beautifulsoup4 transformers torch requests nltk
   ```

## Usage

1. **Scrape Yelp Reviews:**
   - Use the included scraping script to fetch review texts from Yelp. (Make sure to comply with Yelp's terms of service.)

2. **Run the Sentiment Analysis:**
   - Load your scraped data and run the sentiment prediction script. The model will output a sentiment score for each review.

3. **Example:**
   ```python
   import pandas as pd
   from sentiment_analyser import SentimentAnalyser  # Replace with actual import

   # Load reviews
   reviews_df = pd.read_csv('yelp_reviews.csv')
   analyser = SentimentAnalyser()  # Replace with actual class or function
   reviews_df['sentiment'] = reviews_df['review_text'].apply(analyser.predict)

   print(reviews_df[['review_text', 'sentiment']])
   ```

## Libraries Used

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/)
- [transformers](https://huggingface.co/transformers/)
- [torch (PyTorch)](https://pytorch.org/)
- [requests](https://docs.python-requests.org/)
- [nltk](https://www.nltk.org/)

## Notes

- Ensure you have the appropriate permissions to scrape data from Yelp.
- The pretrained BERT model can be swapped or fine-tuned for better accuracy on Yelp-specific data.
- NLTK is used for natural language preprocessing to improve sentiment analysis results.

**Author:** [kamli01](https://github.com/kamli01)
