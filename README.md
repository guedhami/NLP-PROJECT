# Classification of Tweets: Hate Speech and Sentiment Analysis

## Introduction

Social media platforms such as Twitter have become a significant medium for public discourse. However, this open exchange of ideas often leads to the proliferation of hate speech, offensive content, and abusive language. 

This project aims to classify tweets into three categories: **Hate Speech**, **Offensive Language**, and **Neither**, by leveraging machine learning and deep learning techniques. Additionally, linguistic features and sentiment analysis are incorporated to enhance classification performance.

This repository contains the methodologies, results, and insights derived from the project.

## Methodology

### Dataset
The dataset used for this project is sourced from a labeled dataset of tweets. It contains:
- **Tweet Text**: Raw text of tweets.
- **Class Labels**: Categorical labels (**Hate Speech**, **Offensive Language**, **Neither**).

The dataset was preprocessed and split into training and testing sets with an 80:20 ratio.

### Text Preprocessing
To ensure effective feature extraction, tweets were preprocessed using:
- Removal of URLs, mentions, and hashtags.
- Tokenization and stemming using the Porter Stemmer.
- Elimination of stopwords.
- Conversion to lowercase and removal of punctuation.

### Feature Extraction
The following feature representations were used:
1. **TF-IDF Vectorization**: Extracted n-grams (1 to 3) with constraints (`min_df = 5`, `max_df = 0.75`).
2. **POS Tagging**: Part-of-speech tags vectorized using TF-IDF.
3. **Other Linguistic Features**: Sentiment scores (VaderSentiment), readability scores (Flesch-Kincaid), and counts of hashtags, mentions, and URLs.

### Machine Learning Models
The following models were implemented for classification:
1. **Logistic Regression**: A baseline model with L1 and L2 regularization.
2. **LSTM (Long Short-Term Memory)**: A deep learning model leveraging sequential embeddings.
3. **BERT (Bidirectional Encoder Representations from Transformers)**: A transformer model fine-tuned for this classification task.

## Results

### Model Performance Metrics
The performance of each model was evaluated using:
- Precision, Recall, and F1-Score for each class.
- Overall Accuracy.

#### Logistic Regression
```
Hate Speech: Precision=0.75, Recall=0.80, F1-Score=0.77
Offensive Language: Precision=0.83, Recall=0.78, F1-Score=0.80
Neither: Precision=0.85, Recall=0.88, F1-Score=0.86
```

#### LSTM
```
Hate Speech: Precision=0.80, Recall=0.82, F1-Score=0.81
Offensive Language: Precision=0.85, Recall=0.80, F1-Score=0.82
Neither: Precision=0.88, Recall=0.90, F1-Score=0.89
```

#### BERT
```
Hate Speech: Precision=0.88, Recall=0.86, F1-Score=0.87
Offensive Language: Precision=0.90, Recall=0.89, F1-Score=0.89
Neither: Precision=0.92, Recall=0.93, F1-Score=0.92
```

### Model Comparison
| Model                 | Accuracy | Precision (Avg) | F1-Score (Avg) |
|-----------------------|----------|-----------------|----------------|
| Logistic Regression  | 80%      | 81%             | 81%            |
| LSTM                 | 84%      | 85%             | 84%            |
| **BERT**             | **91%**  | **91%**         | **91%**        |

## Explanation of Tools Used

### Libraries
- **Data Processing**: `pandas`, `numpy`
- **Feature Extraction**: `nltk`, `sklearn`, `vaderSentiment`
- **Machine Learning Models**: `sklearn`, `tensorflow`, `transformers`
- **Evaluation and Visualization**: `matplotlib`, `seaborn`

### Deployment Tools
- **Streamlit**: Used to build a web application interface for the project.
- **Ngrok**: Provided a public URL for accessing the Streamlit application remotely.

### Example Screenshots

![Screenshot of Streamlit Application](3.png)

![Screenshot of Ngrok Interface](4.png)

## Discussion

### Key Insights
- **BERT**: Achieved the highest performance, showcasing the strength of transformer-based methods for NLP tasks.
- **LSTM**: Performed well in capturing sequential dependencies in the tweets.
- **Logistic Regression**: Provided a solid baseline when combined with TF-IDF and linguistic features.

### Challenges
- Handling noisy text data (e.g., slang, emojis, abbreviations).
- High computational cost of training models like BERT.
- Addressing class imbalance, as hate speech was underrepresented in the dataset.

## Conclusion and Future Work

This project demonstrated the successful application of machine learning and deep learning models for hate speech classification on Twitter. The **BERT model** achieved the highest accuracy and is recommended for deployment.

### Future Enhancements
- Expand the dataset to improve class balance.
- Explore multilingual models for non-English tweets.
- Develop a real-time web application using `Streamlit` and `Ngrok`.

## References
1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. [VaderSentiment library](https://github.com/cjhutto/vaderSentiment)
3. [Scikit-learn documentation](https://scikit-learn.org/)
4. [TensorFlow documentation](https://www.tensorflow.org/)
