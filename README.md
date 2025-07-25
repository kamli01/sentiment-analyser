# 🎉 Sentiment Analyser for Yelp Reviews 📊

Welcome to the **Sentiment Analyser** project!  
This tool processes textual reviews scraped from Yelp and uses a powerful, pretrained BERT model (with help from NLTK) to classify the sentiment of each review on a scale of **1️⃣ to 5️⃣**.  
Built in Python, it combines modern NLP, deep learning, and web scraping for easy, accurate sentiment analysis!

---

## 🚀 Features

- 🕸️ **Web Scraping:** Extracts review texts from Yelp using BeautifulSoup & Requests.
- 📊 **Data Handling:** Effortless data manipulation with Pandas & Numpy.
- 🧠 **NLP Model:** Utilizes a pretrained BERT model (via HuggingFace Transformers), enhanced with NLTK for robust text preprocessing.
- 🔥 **Deep Learning:** Powered by PyTorch for fast and accurate model inference.
- 🌟 **Sentiment Output:** Scores each review from **1 (very negative) to 5 (very positive)** — see below!

---

## 🌈 Sentiment Output Explained

After processing, each review receives a sentiment rating as a single digit (1-5):

| Rating | Description         | Emoji         |
|--------|---------------------|--------------|
|   1    | Very Negative (Bad) | 😠 👎         |
|   2    | Negative            | 🙁           |
|   3    | Neutral             | 😐           |
|   4    | Positive            | 🙂 👍         |
|   5    | Very Positive (Great) | 🤩 🎉       |

Quickly spot negative, neutral, or highly positive feedback at a glance!

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kamli01/sentiment-analyser.git
   cd sentiment-analyser
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy beautifulsoup4 transformers torch requests nltk
   ```

---

## ⚡ Usage

1. **Scrape Yelp Reviews:**  
   Use the included scraping script to fetch review texts from Yelp.  
   _(Please comply with Yelp's terms of service!)_

2. **Run Sentiment Analysis:**  
   Load your scraped data and run the prediction script. The model outputs a sentiment score for each review.

3. **Example:**
   ```python
   import pandas as pd
   from sentiment_analyser import SentimentAnalyser  # Replace with your actual import

   # Load reviews
   reviews_df = pd.read_csv('yelp_reviews.csv')
   analyser = SentimentAnalyser()  # Replace with actual class or function
   reviews_df['sentiment'] = reviews_df['review_text'].apply(analyser.predict)

   print(reviews_df[['review_text', 'sentiment']])
   ```

---

## 📚 Libraries Used

- 🐼 [pandas](https://pandas.pydata.org/)
- 🟠 [numpy](https://numpy.org/)
- 🍜 [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/)
- 🤗 [transformers](https://huggingface.co/transformers/)
- 🔥 [torch (PyTorch)](https://pytorch.org/)
- 🌐 [requests](https://docs.python-requests.org/)
- 📝 [nltk](https://www.nltk.org/)

---

## 📝 Notes & Tips

- Get permission before scraping data from Yelp.
- The pretrained BERT model can be swapped or fine-tuned for Yelp-specific accuracy.
- NLTK is used for advanced text preprocessing to boost sentiment analysis results.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by [kamli01](https://github.com/kamli01)**
