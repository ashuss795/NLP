# Natural Language Processing (NLP) – 32 Programs

This repository contains **32 Python implementations of Natural Language Processing (NLP) concepts and applications**, covering fundamental text processing techniques, linguistic analysis, feature extraction, machine learning, deep learning, and transformer-based models.

These programs provide hands-on experience with various NLP techniques and libraries, demonstrating how human language can be processed, analyzed, represented, and understood using Python.

---

## About Natural Language Processing

Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables computers to process, analyze, interpret, and generate human language.

Human language is complex because of its grammar, context, ambiguity, vocabulary, and structure. NLP combines computational linguistics, machine learning, and deep learning to enable computers to work with textual and linguistic data.

NLP is widely used in applications such as:

- Chatbots and virtual assistants
- Sentiment analysis
- Machine translation
- Text classification
- Search engines
- Information extraction
- Text summarization
- Speech and language systems
- Document processing

---

## Technologies and Libraries Used

- **Python** – Programming language
- **NLTK** – Natural language processing and text analysis
- **spaCy** – Linguistic processing and entity recognition
- **Scikit-learn** – Machine learning and text feature extraction
- **Gensim** – Word embeddings and semantic analysis
- **NumPy** – Numerical computation
- **TensorFlow / Keras** – Deep learning models
- **PyTorch** – Deep learning framework
- **Hugging Face Transformers** – Pre-trained transformer models
- **TextBlob** – Text processing and sentiment analysis
- **VADER** – Sentiment analysis
- **POT** – Optimal transport computation

---

## List of 32 NLP Programs

| S.No. | Program | Concept |
|---|---|---|
| 1 | Tokenization of Sentences and Words using NLTK and spaCy | Text Preprocessing |
| 2 | Stemming and Lemmatization on Sample Text | Text Normalization |
| 3 | Stop-word Removal from a Document | Text Preprocessing |
| 4 | Part-of-Speech (POS) Tagging | Linguistic Analysis |
| 5 | Parsing and Chunking using RegEx and spaCy | Syntax Analysis |
| 6 | Named Entity Recognition (NER) using spaCy | Entity Recognition |
| 7 | Coreference Resolution | Linguistic Analysis |
| 8 | Bag-of-Words (BoW) Vectorization and Representation | Feature Extraction |
| 9 | TF-IDF Implementation and Comparison with BoW | Feature Extraction |
| 10 | N-Gram Model Generation | Language Modeling |
| 11 | Cosine Similarity between Text Documents | Text Similarity |
| 12 | Word2Vec Word Embeddings using Gensim | Word Embeddings |
| 13 | GloVe Embeddings Loading and Vector Representation | Word Embeddings |
| 14 | Text Similarity using Word Mover's Distance (WMD) | Semantic Similarity |
| 15 | Text Classification using Naïve Bayes/SVM with TF-IDF | Machine Learning |
| 16 | Sentiment Analysis using TextBlob and VADER | Sentiment Analysis |
| 17 | Topic Modeling using Latent Dirichlet Allocation (LDA) | Topic Modeling |
| 18 | Topic Modeling using Latent Semantic Analysis (LSA) | Topic Modeling |
| 19 | Opinion Mining on Product/Service Reviews | Opinion Mining |
| 20 | Information Extraction from Structured and Unstructured Documents | Information Extraction |
| 21 | Information Retrieval System with Ranking using TF-IDF | Information Retrieval |
| 22 | Sequence Classification using RNN/LSTM | Deep Learning |
| 23 | Implementing a Basic Attention Mechanism | Deep Learning |
| 24 | Fine-Tuning BERT for Text Classification using Hugging Face | Transformers |
| 25 | Sentence Embeddings using BERT and GPT-2 | Transformers |
| 26 | Text Generation using GPT-2 | Text Generation |
| 27 | Named Entity Recognition using Transformers (BERT) | Transformers |
| 28 | Machine Translation using MarianMT or T5 | Machine Translation |
| 29 | Text Summarization using BART or T5 | Text Summarization |
| 30 | Chatbot Development using a Pre-trained Transformer | Conversational NLP |
| 31 | Automatic Document Classification and Separation using ML | Document Classification |
| 32 | Hybrid Probabilistic and FSM-Based Sequence Modeling for Document Boundary Detection | Sequence Modeling |

---

## Detailed Description of Programs

### 1. Tokenization of Sentences and Words using NLTK and spaCy

Demonstrates tokenization using NLTK and spaCy. Tokenization is the process of breaking text into smaller units such as sentences and words for further processing.

**Libraries:** NLTK, spaCy

### 2. Stemming and Lemmatization on Sample Text

Demonstrates stemming and lemmatization. Stemming reduces words to their root form by removing prefixes or suffixes, while lemmatization converts words into their meaningful dictionary forms.

**Library:** NLTK

### 3. Stop-word Removal from a Document

Removes common English stop words such as "the", "is", "and", and "of" from text to help reduce unnecessary words during text processing.

**Library:** NLTK

### 4. Part-of-Speech (POS) Tagging

Assigns grammatical categories to words based on their context in a sentence, including nouns, verbs, adjectives, adverbs, pronouns, and prepositions.

**Library:** NLTK

### 5. Parsing and Chunking using RegEx and spaCy

Demonstrates parsing and chunking using regular expressions and spaCy. Parsing analyzes grammatical structures, while chunking groups related words into meaningful phrases.

**Libraries:** spaCy, RegEx

### 6. Named Entity Recognition (NER) using spaCy

Identifies and classifies named entities such as people, organizations, locations, dates, monetary values, and other entities from text.

**Library:** spaCy

### 7. Coreference Resolution

Identifies expressions in a document that refer to the same entity, such as linking a person's name with pronouns referring to that person.

**Libraries:** spaCy, neuralcoref or similar tools

### 8. Bag-of-Words (BoW) Vectorization and Representation

Converts text documents into numerical vectors based on word occurrence counts. This representation is used as input for various machine learning algorithms.

**Library:** Scikit-learn

### 9. TF-IDF Implementation and Comparison with BoW

Implements Term Frequency-Inverse Document Frequency (TF-IDF) and compares it with Bag-of-Words. TF-IDF assigns weights to words based on their frequency and importance across a collection of documents.

**Library:** Scikit-learn

### 10. N-Gram Model Generation

Generates unigrams, bigrams, and trigrams from a text corpus. Unigrams represent one word, bigrams two consecutive words, and trigrams three consecutive words.

**Libraries:** NLTK, Scikit-learn

### 11. Cosine Similarity between Text Documents

Calculates similarity between text documents by comparing the angle between their numerical vector representations.

**Libraries:** Scikit-learn, NumPy

### 12. Word2Vec Word Embeddings using Gensim

Trains a Word2Vec model on a text corpus to generate numerical word embeddings that capture semantic and contextual relationships between words.

**Library:** Gensim

### 13. GloVe Embeddings Loading and Vector Representation

Loads pre-trained GloVe embeddings and represents words as numerical vectors that capture semantic relationships based on their usage in a large text corpus.

**Library:** Gensim

### 14. Text Similarity using Word Mover's Distance (WMD)

Calculates semantic similarity between documents using Word Mover's Distance, which measures the distance required to transform one word distribution into another.

**Libraries:** Gensim, POT

### 15. Text Classification using Naïve Bayes/SVM with TF-IDF

Classifies text documents into predefined categories using machine learning algorithms such as Naïve Bayes or Support Vector Machine (SVM) with TF-IDF feature extraction.

**Library:** Scikit-learn

### 16. Sentiment Analysis using TextBlob and VADER

Analyzes textual data to determine whether the sentiment is positive, negative, or neutral and calculates sentiment scores.

**Libraries:** TextBlob, VADER Sentiment

### 17. Topic Modeling using Latent Dirichlet Allocation (LDA)

Applies Latent Dirichlet Allocation to discover hidden topics in a collection of documents by identifying groups of related words.

**Library:** Scikit-learn

### 18. Topic Modeling using Latent Semantic Analysis (LSA)

Uses Latent Semantic Analysis with Singular Value Decomposition (SVD) to identify underlying semantic patterns and relationships in textual data.

**Libraries:** Scikit-learn, NumPy

### 19. Opinion Mining on Product/Service Reviews

Analyzes product or service reviews to extract opinions and determine whether customer feedback is positive, negative, or neutral.

**Libraries:** TextBlob, NLTK

### 20. Information Extraction from Structured and Unstructured Documents

Extracts useful information such as names, organizations, locations, dates, and other relevant entities from structured and unstructured textual data.

**Libraries:** spaCy, Regular Expressions

### 21. Information Retrieval System with Ranking using TF-IDF

Implements an information retrieval system that searches documents based on user queries and ranks results using TF-IDF and cosine similarity.

**Library:** Scikit-learn

### 22. Sequence Classification using RNN/LSTM

Demonstrates sequence classification using Recurrent Neural Networks (RNN) or Long Short-Term Memory (LSTM) networks to learn patterns in sequential textual data.

**Libraries:** TensorFlow, Keras

### 23. Implementing a Basic Attention Mechanism

Implements a basic attention mechanism that enables neural networks to focus on relevant parts of an input sequence during processing.

**Libraries:** TensorFlow, Keras

### 24. Fine-Tuning BERT for Text Classification using Hugging Face

Demonstrates fine-tuning a pre-trained BERT model for text classification by adapting the model to a specific task using labeled data.

**Libraries:** Hugging Face Transformers, PyTorch

### 25. Sentence Embeddings using BERT and GPT-2

Generates numerical representations of sentences using pre-trained BERT and GPT-2 models for semantic representation and similarity-related tasks.

**Libraries:** Hugging Face Transformers, PyTorch

### 26. Text Generation using GPT-2

Demonstrates text generation using the pre-trained GPT-2 language model, which generates additional text based on a given input prompt.

**Library:** Hugging Face Transformers

### 27. Named Entity Recognition using Transformers (BERT)

Performs named entity recognition using a pre-trained transformer model based on BERT to identify entities such as person names, organizations, and locations.

**Library:** Hugging Face Transformers

### 28. Machine Translation using MarianMT or T5

Demonstrates machine translation using pre-trained transformer models such as MarianMT or T5 to convert text from one language to another.

**Library:** Hugging Face Transformers

### 29. Text Summarization using BART or T5

Performs automatic text summarization using transformer models such as BART or T5 to generate concise versions of documents while preserving their important information.

**Library:** Hugging Face Transformers

### 30. Chatbot Development using a Pre-trained Transformer

Demonstrates chatbot development using a pre-trained conversational transformer such as DialoGPT to generate responses based on user input.

**Library:** Hugging Face Transformers

### 31. Automatic Document Classification and Separation using ML

Classifies documents into predefined categories using machine learning techniques and organizes or separates documents according to their predicted category.

**Libraries:** Scikit-learn, NumPy

### 32. Hybrid Probabilistic and FSM-Based Sequence Modeling for Document Boundary Detection

Demonstrates document boundary detection using probabilistic techniques combined with a Finite State Machine (FSM). It identifies document sections or boundaries using textual patterns and state-transition rules.

**Libraries:** Python, Regular Expressions

---

## Learning Objectives

The primary learning objectives of these 32 NLP programs are:

- **Text Preprocessing:** Learn tokenization, stemming, lemmatization, stop-word removal, and text normalization.
- **Linguistic Analysis:** Understand POS tagging, parsing, chunking, named entity recognition, and coreference resolution.
- **Text Representation:** Explore Bag-of-Words, TF-IDF, N-Grams, Word2Vec, and GloVe.
- **Text Similarity:** Understand cosine similarity and Word Mover's Distance.
- **Machine Learning:** Implement text classification, opinion mining, sentiment analysis, and information retrieval.
- **Topic Modeling:** Explore LDA and LSA for discovering topics in textual data.
- **Deep Learning:** Learn sequence classification, RNN, LSTM, and attention mechanisms.
- **Transformer Models:** Explore BERT, GPT-2, MarianMT, BART, and T5.
- **Practical Applications:** Develop understanding of chatbots, document classification, translation, summarization, and document boundary detection.

---

## How to Run the Programs

1. Clone or download this repository.
2. Open the required `.ipynb` notebook using Google Colab, Jupyter Notebook, or VS Code.
3. Install the required Python libraries.
4. Download any required datasets or pre-trained language models.
5. Execute the notebook cells sequentially.
6. Observe the outputs and results of each program.

---

## Programming Language

**Python**

## Environment

**Jupyter Notebook / Google Colab / VS Code**

---

## Author

**Ashutosh Shukla**

---

*This repository is intended for educational purposes and practical learning of Natural Language Processing concepts and applications.*
