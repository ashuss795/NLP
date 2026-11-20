# Natural Language Processing (NLP) Experiments

A collection of **Natural Language Processing (NLP) practicals**
implemented in Python. The experiments cover text preprocessing,
numerical text representation, word embeddings, text similarity, machine
learning, sequence modeling, and Transformer-based NLP applications.

The notebooks use tools and libraries including **NLTK, spaCy,
scikit-learn, Gensim, TextBlob, VADER, TensorFlow, PyTorch, and Hugging
Face Transformers**. They are intended for learning and experimentation
in Jupyter Notebook or Google Colab.

## Table of Contents

-   [Experiments Included](#experiments-included)
-   [Technologies and Libraries](#technologies-and-libraries)
-   [Getting Started](#getting-started)
-   [Running the Notebooks](#running-the-notebooks)
-   [Learning Outcomes](#learning-outcomes)
-   [Important Notes](#important-notes)
-   [Author](#author)
-   [License](#license)

## Experiments Included

The three supplied README files contain **31 distinct experiment
descriptions** after removing repeated entries. The source files refer
to a 32-practical collection, but one distinct experiment is not
identifiable from the supplied descriptions. The catalogue below
includes every distinct experiment described in those files without
inventing a missing program.


## Quick Links to the 32 Programs

Click any program name below to open its notebook directly in GitHub or your Markdown viewer.

| No. | Program (click to open) |
|---:|---|
| 1 | [Tokenization of Sentences and Words using NLTK and spaCy](./Tokenization.ipynb) |
| 2 | [Stemming and Lemmatization on Sample Text](./Stemming_and_Lemmatization.ipynb) |
| 3 | [Stop-word Removal from a Document](./Stop_word_Removal.ipynb) |
| 4 | [Part-of-Speech (POS) Tagging](./Part_of_Speech_%28POS%29_Tagging.ipynb) |
| 5 | [Parsing and Chunking using RegEx and spaCy](./Parsing_and_Chunking.ipynb) |
| 6 | [Named Entity Recognition (NER) using spaCy](./Named_Entity_Recognition.ipynb) |
| 7 | [Coreference Resolution](./Coreference_Resolution_using_neuralcoref.ipynb) |
| 8 | [Bag-of-Words (BoW) Vectorization and Representation](./Bag_of_Words_%28BoW%29.ipynb) |
| 9 | [TF-IDF Implementation and Comparison with BoW](./TF_IDF_Implementation_and_Comparison.ipynb) |
| 10 | [N-Gram Model Generation](./N_Gram_Model_%28Uni_%2C_Bi_%2C_Tri_gram%29.ipynb) |
| 11 | [Cosine Similarity between Text Documents](./Cosine_Similarity_Computation.ipynb) |
| 12 | [Word2Vec Word Embeddings using Gensim](./Word2Vec_Word_Embeddings_using_Gensim_on_a_Custom_Corpus.ipynb) |
| 13 | [GloVe Embeddings Loading and Vector Representation](./GloVe_Embeddings_Loading_and_Vector_Representation.ipynb) |
| 14 | [Text Similarity using Word Mover's Distance (WMD)](./Text_Similarity_using_Word_Mover%27s_Distance_%28WMD%29.ipynb) |
| 15 | [Text Classification using Naïve Bayes/SVM with TF-IDF](./Text_Classification_using_Na%C3%AFve_Bayes_SVM_with_TF_IDF.ipynb) |
| 16 | [Sentiment Analysis using TextBlob and VADER](./Sentiment_Analysis_using_TextBlob_and_VADER.ipynb) |
| 17 | [Topic Modeling using Latent Dirichlet Allocation (LDA)](./Topic_Modeling_using_Latent_Dirichlet_Allocation_%28LDA%29.ipynb) |
| 18 | [Topic Modeling using Latent Semantic Analysis (LSA)](./Topic_Modeling_using_Latent_Semantic_Analysis_%28LSA%29.ipynb) |
| 19 | [Opinion Mining on Product/Service Reviews](./Opinion_mining_on_product_service_reviews_dataset.ipynb) |
| 20 | [Information Extraction from Structured and Unstructured Documents](./Information_Extraction_%28IE%29_from_Structured_Unstructured_Documents.ipynb) |
| 21 | [Information Retrieval System with Ranking using TF-IDF](./Information%20Retrieval%20System%20with%20Ranking%20using%20TF-IDF.ipynb) |
| 22 | [Sequence Classification using RNN/LSTM](./Sequence_Classification_using_RNN_LSTM_%28Keras_TensorFlow%29.ipynb) |
| 23 | [Implementing a Basic Attention Mechanism](./Implementing_Attention_Mechanism_%28Basic_custom_model%29.ipynb) |
| 24 | [Fine-Tuning BERT for Text Classification using Hugging Face](./Fine_Tuning_BERT_for_text_classification_using_Hugging_Face.ipynb) |
| 25 | [Sentence Embeddings using BERT and GPT-2](./Sentence_Embeddings_using_BERT_and_GPT_2.ipynb) |
| 26 | [Text Generation using GPT-2](./Text_Generation_using_GPT_2_on_a_custom_dataset.ipynb) |
| 27 | [Named Entity Recognition using Transformers (BERT)](./Named%20Entity%20Recognition%20using%20Transformers%28BERT%29.ipynb) |
| 28 | [Machine Translation using MarianMT or T5](./Machine%20Translation%20using%20MarianMT%20or%20T5%20%28Hugging%20Face%29.ipynb) |
| 29 | [Text Summarization using BART or T5](./Text%20Summarization%20using%20BART%20or%20T5.ipynb) |
| 30 | Chatbot Development using a Pre-trained Transformer — **not included as a notebook in this ZIP** |
| 31 | [Automatic Document Classification and Separation using ML](./Automatic%20Document%20Classification%20and%20Separation%20using%20ML.ipynb) |
| 32 | [Hybrid Probabilistic and FSM-Based Sequence Modeling for Document Boundary Detection](./Hybrid%20Probabilistic%20%2B%20FSM-Based%20Sequence%20Modeling%20for%20document%20boundary%20detection.ipynb) |

> **Note:** The ZIP contains 31 Jupyter notebooks, not 32. Program 30 (Chatbot Development using a Pre-trained Transformer) is listed in the program catalogue but its `.ipynb` file is missing. Add that notebook to this folder and then add its relative link here.

### 1. NLP Fundamentals and Text Preprocessing

  -----------------------------------------------------------------------------
                           No. Experiment            Description
  ---------------------------- --------------------- --------------------------
                            01 Tokenization of       Splits input text into
                               Sentences and Words   sentences and individual
                                                     words using NLTK and
                                                     spaCy.

                            02 Stemming and          Compares stemming with
                               Lemmatization         lemmatization to reduce
                                                     words to root-like or
                                                     meaningful base forms.

                            03 Stop-word Removal     Removes commonly occurring
                                                     English stop words from
                                                     text using NLTK.

                            04 Part-of-Speech        Assigns grammatical labels
                               Tagging               such as noun, verb,
                                                     adjective, adverb, and
                                                     determiner to words.

                            05 Parsing and Chunking  Demonstrates
                               using RegEx and spaCy regular-expression-based
                                                     chunking, dependency
                                                     parsing, and noun-chunk
                                                     extraction.

                            06 Named Entity          Identifies entities such
                               Recognition           as people, organizations,
                                                     locations, and dates using
                                                     spaCy.
  -----------------------------------------------------------------------------

### 2. Text Representation, Similarity, and Embeddings

  ------------------------------------------------------------------------
                           No. Experiment            Description
  ---------------------------- --------------------- ---------------------
                            07 Bag-of-Words (BoW)    Converts documents
                               Vectorization and     into numerical
                               Representation        vectors based on word
                                                     frequency using
                                                     CountVectorizer.

                            08 TF-IDF Implementation Implements TF-IDF and
                               and Comparison with   compares its
                               BoW                   representation with
                                                     Bag-of-Words.

                            09 N-Gram Model          Generates unigram,
                               Generation            bigram, and trigram
                                                     sequences from a text
                                                     corpus.

                            10 Cosine Similarity     Converts documents
                               between Text          into TF-IDF vectors
                               Documents             and measures their
                                                     cosine similarity.

                            11 Word2Vec Embeddings   Creates word
                                                     embeddings using
                                                     Gensim on a custom
                                                     corpus.

                            12 GloVe Embeddings      Loads GloVe
                                                     embeddings and
                                                     represents words as
                                                     vectors.

                            13 Text Similarity using Measures semantic
                               Word Mover's Distance similarity between
                               (WMD)                 text documents using
                                                     Word Mover's
                                                     Distance.
  ------------------------------------------------------------------------

### 3. Machine Learning and Text Analysis

  ------------------------------------------------------------------------
                           No. Experiment            Description
  ---------------------------- --------------------- ---------------------
                            14 Text Classification   Performs text
                               using Naive Bayes and classification using
                               SVM with TF-IDF       Naive Bayes, SVM, and
                                                     TF-IDF features.

                            15 Sentiment Analysis    Analyzes the
                               using TextBlob and    sentiment expressed
                               VADER                 in text using
                                                     TextBlob and VADER.

                            16 Topic Modeling using  Discovers latent
                               Latent Dirichlet      topics in a document
                               Allocation (LDA)      collection using LDA.

                            17 Topic Modeling using  Explores topic
                               Latent Semantic       structure using
                               Analysis (LSA)        Latent Semantic
                                                     Analysis.

                            18 Opinion Mining on     Analyzes opinions
                               Product and Service   expressed in product
                               Reviews               and service reviews.

                            19 Information           Extracts relevant
                               Extraction from       information from
                               Structured and        structured and
                               Unstructured          unstructured document
                               Documents             content.

                            20 Information Retrieval Builds a document
                               System with TF-IDF    retrieval system that
                               Ranking               ranks results using
                                                     TF-IDF.

                            21 Sequence              Demonstrates sequence
                               Classification using  classification with
                               RNN and LSTM          RNN and LSTM models
                                                     using Keras and
                                                     TensorFlow.

                            22 Implementing an       Implements a basic
                               Attention Mechanism   custom attention
                                                     mechanism.
  ------------------------------------------------------------------------

### 4. Transformer-Based and Advanced NLP

  ------------------------------------------------------------------------
                           No. Experiment            Description
  ---------------------------- --------------------- ---------------------
                            23 Fine-Tuning BERT for  Fine-tunes a
                               Text Classification   pretrained BERT model
                                                     for text
                                                     classification using
                                                     Hugging Face.

                            24 Sentence Embeddings   Generates sentence
                               using BERT and GPT-2  representations using
                                                     Transformer-based
                                                     models.

                            25 Text Generation using Generates text using
                               GPT-2                 GPT-2 on a custom
                                                     dataset.

                            26 Named Entity          Identifies entities
                               Recognition using     such as names,
                               Transformers (BERT)   locations, and
                                                     organizations using a
                                                     Transformer-based
                                                     model.

                            27 Machine Translation   Performs machine
                               using MarianMT or T5  translation using
                                                     pretrained Hugging
                                                     Face models.

                            28 Text Summarization    Generates concise
                               using BART or T5      summaries from longer
                                                     text documents.

                            29 Text Summarization    Explores an
                               using BART or T5      additional
                                                     implementation of
                                                     Transformer-based
                                                     text summarization.

                            30 Automatic Document    Classifies documents
                               Classification and    and separates them
                               Separation using ML   into categories using
                                                     machine learning.

                            31 Hybrid                Combines
                               Probabilistic +       probabilistic methods
                               FSM-Based Sequence    and Finite State
                               Modeling              Machines (FSM) for
                                                     document-boundary
                                                     detection and
                                                     sequence modeling.
  ------------------------------------------------------------------------

> **Catalogue note:** The source README for the advanced experiments
> labels these final nine entries as 24--32. The consolidated catalogue
> renumbers the distinct descriptions sequentially. The supplied files
> do not provide a separate description for a 32nd unique experiment
> beyond the hybrid probabilistic + FSM-based sequence-modeling entry.

## Technologies and Libraries

The experiments use the following tools and libraries, depending on the
notebook:

-   **Language:** Python
-   **Notebook environments:** Jupyter Notebook, Google Colab
-   **NLP:** NLTK, spaCy, TextBlob, VADER Sentiment
-   **Machine Learning:** scikit-learn
-   **Numerical and data processing:** NumPy, pandas
-   **Word embeddings:** Gensim, Word2Vec, GloVe
-   **Deep Learning:** TensorFlow, Keras, PyTorch
-   **Transformers and pretrained models:** Hugging Face Transformers,
    BERT, GPT-2, BART, T5, MarianMT
-   **Visualization:** Matplotlib

## Getting Started

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your actual GitHub repository URL and
local folder name.

### 2. Create a virtual environment (recommended)

``` bash
python -m venv .venv
```

Activate it:

**Windows**

``` bash
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

### 3. Install the commonly used dependencies

``` bash
pip install jupyter numpy pandas scikit-learn nltk spacy gensim textblob vaderSentiment matplotlib tensorflow torch transformers sentencepiece
```

Some notebooks may require additional packages, datasets, pretrained
embeddings, language resources, or model-specific dependencies. Refer to
the setup cells and instructions in the relevant notebook.

### 4. Launch Jupyter Notebook

``` bash
jupyter notebook
```

Alternatively, open the repository in JupyterLab or upload the notebooks
to Google Colab.

## Running the Notebooks

1.  Open the notebook for the experiment you want to run (`.ipynb`).
2.  Read its introductory notes and check for dataset or model
    requirements.
3.  Install any notebook-specific dependencies.
4.  Configure dataset paths or other environment-specific settings where
    needed.
5.  Run the cells sequentially.

Some notebooks download pretrained models or language resources when
executed. An internet connection may be required for those downloads.

## Learning Outcomes

By working through these experiments, you can:

-   Apply text preprocessing techniques, including tokenization,
    stemming, lemmatization, and stop-word removal.
-   Perform part-of-speech tagging, parsing, chunking, and named entity
    recognition.
-   Convert text into numerical representations using BoW, TF-IDF, and
    N-Grams.
-   Explore word embeddings and semantic similarity techniques.
-   Build text classification, sentiment analysis, topic modeling, and
    opinion mining workflows.
-   Implement information extraction and information retrieval
    techniques.
-   Understand sequence modeling with RNNs, LSTMs, and attention.
-   Explore Transformer-based NLP tasks such as text generation,
    translation, summarization, and classification.

## Important Notes

-   Run notebook cells in the order provided unless the notebook
    specifies otherwise.
-   Check each notebook for its own dataset, package, and
    language-resource requirements.
-   Update dataset paths to match your local environment.
-   Some experiments use pretrained embeddings or download model weights
    at runtime.
-   Transformer models can require substantial RAM, GPU memory, and
    download time.
-   Runtime and hardware requirements depend on the model and dataset
    used by each notebook.

## Author

**Ashutosh Shukla**

## License

These practicals are intended for educational and learning purposes. No
specific open-source license is identified in the supplied README files.
If you plan to distribute or encourage reuse of this repository, add an
appropriate license file and update this section accordingly.
