# Twitter-Sentiment-Analysis-NLP-Pipeline
Descriptive Report on Sentiment Analysis Using Different Vectorization Methods 

Methodology: 

In this sentiment analysis project, various vectorization techniques were used to evaluate model performance on a dataset of social media text.  

Python Libraries Used: - 

pandas: For data manipulation and analysis in a structured format (DataFrames). 

numpy: For numerical computations. 

re: For working with regular expressions. 

nltk: Natural Language Toolkit, used for text preprocessing, tokenization, and lemmatization. 

unidecode: For converting Unicode characters to ASCII, ensuring uniformity in text. 

contractions: For expanding contractions in the text during preprocessing. 

spacy: Advanced NLP library for text processing, tokenization, and lemmatization. 

WordNetLemmatizer (from nltk): For lemmatizing words, reducing them to their base form. 

word_tokenize (from nltk): For tokenizing sentences into words. 

train_test_split (from sklearn): For splitting the dataset into training and testing sets. 

SelectKBest (from sklearn): For feature selection using statistical tests. 

chi2 (from sklearn): Chi-square test for selecting features based on importance. 

TextBlob: For simple NLP tasks like sentiment analysis and text processing. 

LabelEncoder (from sklearn): For encoding target labels into numeric values. 

CountVectorizer (from sklearn): For converting text documents into a matrix of token counts (Bag of Words). 

TfidfVectorizer (from sklearn): For transforming text into TF-IDF feature vectors. 

accuracy_score, classification_report, confusion_matrix (from sklearn): For evaluating the performance of machine learning models. 

LogisticRegression (from sklearn): For implementing a logistic regression model for classification. 

tensorflow_hub: For using a pre-trained model Universal Sentence Encoder. 

gensim.downloader: For loading pre-trained word embeddings (Word2Vec, etc.) from the Gensim library. 

concurrent.futures: For parallel processing to speed up computations. 

svm (from sklearn): Support Vector Machine algorithm for classification tasks. 

The following steps detail the preprocessing, vectorization, and model implementation used: 

Preprocessing: 

Data Cleaning: All text data was preprocessed by removing unnecessary characters, punctuation, and stopwords. Lowercasing was applied to ensure uniformity. Steps: 

Dropped unwanted columns 

Removal of URLs, punctuations, mentions, hashtags, special characters, and emojis. 

Removal of accents like é --> e. 

Expanded words like wouldn’t --> would not 

Removal of stopwords 

For this initially, I tried extending the stopword keywords to the frequently occurring words and tried but the accuracy was reduced at this try. So basic stopword removal from the library along with a brief list of custom stopwords is what I did. 

Lemmatization – I used lemmatization for taking root words because stemming is giving unfinished words sometimes. 

I used the spacy model named “en_core_web_sm” for the lemmatization, when I tried with nltk WordNetLemmatizer I was not getting accurate results, which is why I used the spacy model. 

Vectorization Methods: 

Bag of Words (BoW): The frequency of each word in the document is counted to create a feature vector. I used CountVectorizer() for this. 

TF-IDF (Term Frequency - Inverse Document Frequency): This method adjusts word counts based on how frequently words appear across documents, giving more weight to less common but informative words. Here I used TfidfVectorizer() for this. 

Word2Vec: This embedding-based method captures semantic meaning by creating continuous vector representations of words. word2vec-google-news-300 is the model I used which is trained using the dataset of about 100 billion words from Google News articles 

GloVe (Global Vectors for Word Representation): Another word embedding technique, which focuses on aggregating global word-word co-occurrence statistics to create vector representations. I used the glove-wiki-gigaword-100 model which is trained with 2 main Corporas which are Wikipedia and Gigaword. To get this model I used the Gensim library. 

Feature Selection: Chi-square feature selection was used to identify the most significant features from the text after vectorization. 

Bonus Part: I have been using Universal Sentence Encoder from TensorFlow Hub. Unlike the word embedding techniques in which we represent words into vectors, in Sentence Embeddings entire sentence or text along with its semantics information is mapped into vectors of real numbers. For doing this, since it needs the whole sentence and words I have compared with and without pre-processing in the Findings part. 

Model Implementation: 

SVM was chosen as the primary model to evaluate the performance of different vectorization techniques. The model was trained using vectorized data and evaluated for accuracy and other performance metrics. 

Logistic Regression is used as a secondary model to see the difference between the models. 

Findings: 

The following results summarize the performance of each vectorization method based on accuracy: 

**Bag of Words (BoW)**: 
Accuracy: 71.58% 

Performance Discussion:  

BoW performed well with following all pre-processing steps which are Basic cleaning, stopword removal, and lemmatization. 

BoW showed a good balance between precision, recall, and F1-score across all sentiment classes (Negative, Neutral, Positive).  

However, as expected, the simplicity of BoW led to some loss of semantic context, which could explain misclassifications for more complex sentiment expressions. 

 

**TF-IDF with Chi-Square Feature Selection**: 
Highest Accuracy Attained: 72.95% 

Performance Discussion:  

Unlike BoW TF-IDF with basic cleaning, and lemmatization gave more accuracy. 

TF-IDF with feature selection improved slightly over BoW. By focusing on more informative words.  

However, it still struggled with capturing deeper semantic meaning in the text. 

 

**Word2Vec Embeddings**: 
Highest Accuracy Obtained: 69.5% 

Performance Discussion:  

Word2Vec embeddings introduced semantic understanding into the model, resulting in better contextual accuracy, particularly for Positive sentiments.  

However, it still underperformed compared to BoW and TF-IDF, likely because the embeddings needed further fine-tuning or training. 

**GloVe Embeddings**: 
Accuracy: 63.06% 

Performance Discussion:  

GloVe embeddings provided lower accuracy than other methods. While GloVe captures global co-occurrences well, it might not have been the best fit for the dataset. 

 

**Universal Sentence Encoder (USE)**: 
Accuracy: 68.22% 

Performance Discussion:  

In USE we can give the entire text sentences to the embedding, so I have tried with and without preprocessing steps. Among them I got the highest accuracy of 68.22% for basic cleaning without stopword removal and lemmatization. 

This suggests that USE benefits from maintaining the sentence structure and context, as removing stopwords and altering word forms through lemmatization may have removed crucial contextual information. 

Comparison Summary: 

TF-IDF with Chi-Square Selection appeared as the best-performing method in terms of accuracy, likely because it focuses on the most informative features. 

BoW performed well, showing that simpler models can still achieve competitive results for certain tasks. 

Word2Vec and GloVe embeddings, though promising for their semantic representation, did not outperform the traditional methods. This could be attributed to insufficient fine-tuning or the size and specificity of the dataset. 

From the results, it is clear that TF-IDF outperformed other methods, with Logistic Regression achieving the highest accuracy of 73.36%. The SVM model also performed well with TF-IDF, reaching 72.95%, confirming that this method effectively captures important textual features for classification. 

Bag-of-Words (BoW) closely followed, with both SVM and Logistic Regression models delivering competitive accuracy scores above 71%. This demonstrates that simpler models can still achieve competitive results for certain tasks. 

Word2Vec performed slightly lower, indicating that while it captures semantic meaning, it might require more complex tuning or larger datasets to achieve results for achieving higher accuracy. 

GloVe had the lowest performance in both SVM and Logistic Regression models, with accuracy scores hovering around 62-63%. This suggests that GloVe embeddings, although useful in some cases, might not have been the best fit for this or it might need to be tuned finely or trained with larger datasets. 

Finally, while capturing sentence-level information, Universal Sentence Encoder (USE) did not surpass TF-IDF or BoW in terms of accuracy. It still provided reasonable results, especially in SVM (68.22%) but could benefit from additional fine-tuning or preprocessing adjustments. 

In summary, TF-IDF stands out as the most effective vectorization method for this sentiment analysis task, especially when combined with Logistic Regression. At the same time, simpler methods like BoW still offer competitive results with minimal preprocessing. 
