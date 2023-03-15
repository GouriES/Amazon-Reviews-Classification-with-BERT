# Amazon-Reviews-Classification-with-BERT
BERT is a very powerful language representation model that has made a significant contribution to the field of natural language processing (NLP). It has made it much easier to perform transfer learning in NLP.

Amazon Reviews Classification with BERT
Build a binary classification model based on the amazon reviews.
Bidirectional Encoder Representations from Transformers
BERT stands for Bidirectional Encoder Representations from Transformers. It is a powerful pre-trained language model developed by Google in 2018 that uses deep learning techniques to understand the context of words in a sentence.
Table of Contents:
1. Introduction
2. Business Problem
3. Dataset Column Analysis
4. Data Preprocessing
5. Exploratory Data Analysis
6. Creating BERT Model
7. Tokenization
8. Training a NN with 768 features
9. Creating a Data pipeline for BERT Model
10. Observations
11. Reference
## Introduction
In this blog we are using the amazon reviews dataset from kaggle for binary classification. Amazon reviews are a popular dataset used in natural language processing (NLP) research and applications. Amazon is one of the largest e-commerce platforms in the world, and as such, it has a massive collection of customer reviews for a wide variety of products.
## Business Problem
BERT classification models need a lot of high-quality labeled data. However, in some sectors, such as legal and healthcare, such data may be hard to come by or cost a lot to obtain. Additionally, bias or errors in the data may have a negative impact on the model's accuracy. Training and running BERT classification models necessitate a significant investment in processing power and memory due to their computationally demanding nature. Businesses with limited computing resources or those who cannot afford costly hardware upgrades may find this to be a significant constraint.
## Dataset Column Analysis
Source of Data: The dataset is given on Kaggle's website. Please find the link below.
Amazon Fine Food Reviews
Analyze ~500,000 food reviews from Amazonwww.kaggle.com
The context of this dataset is Amazon reviews of fine foods. The information spans more than ten years and includes all 500,000 reviews as of October 2012. Ratings, information about the product and its users, and a review in plain text are all part of a review. Reviews from all other Amazon categories are also included.
Data includes:
Reviews from Oct 1999 - Oct 2012
568,454 reviews
256,059 users
74,258 products
260 users with > 50 reviews
## Data Preprocessing
First, create some features in the datasets.
In the reviews.csv file contains columns i.e Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text from these features we have to choose the relevant one which is texts, score.
After loading the reviews data, the values are evaluated. If the score is greater than 3, the value is 1, and otherwise it is 0, and the None values map to the score.
Splitting text data and removing HTML from the Text column using regular expressions
After dividing the data into train and test, we can see that the counts of score for train and test data are distributed equally when the bar graph is plotted.

## Exploratory Data Analysis
It is critical to ensure that both the train and test sets are representative of the entire dataset when splitting data. As a result, the target variable's (the score's) distribution ought to be comparable in both sets.
Plot the count of scores in each set on a bar graph to compare their distributions after splitting the data and creating the train and test sets. The split was successful and both sets are representative of the entire dataset if the counts are roughly equal. However, it is essential to keep in mind that the count of scores on its own may not provide a complete picture of the target variable's distribution. In order to guarantee that the two sets are comparable, additional aspects like the range and variability of scores should be taken into consideration.

## Creating BERT Model
In the bert model, we construct mask vectors, segment vectors, and a word sequence with integer Word_ids. When using BERT to conduct sentiment analysis on a dataset, each input sequence is represented by a callable object that can be used as a Keras layer. The sequence of hidden states (embeddings) at the output of the final layer of the BERT model is called sequence output. The [CLS] token embedding is a part of it. The [CLS] token embedded from Sequence output is the pooled output, which is further processed by a linear layer and a Tanh activation function. During pre training, the next sentence prediction (classification) objective is used to train the linear layer weights.

## Tokenization
BERT's tokenizer makes a sequence of tokens out of the raw text, with each token representing a subword or workpiece. The process of breaking down a word into smaller units known as subwords is called subword tokenization. By allowing the tokenizer to divide subwords into even smaller units based on the corpus frequency, wordpiece tokenization takes subwords one step further.
BERT's tokenizer is made to work with tokenization of wordpieces and subwords to create a standard format for input text. In order for the neural network to efficiently process the input, this standard format is used to ensure that the input sequence has a fixed length.
Special tokens like [CLS] and [SEP], which are used to denote the beginning and end of a sentence, are also handled by the BERT tokenizer.
In general, the BERT tokenizer is an essential component of the BERT model that is responsible for transforming raw text into a format that is usable by the neural network for processing

## Getting Embeddings from BERT Model
Tokenization is the process of breaking down a longer piece of text into smaller units or tokens, which can then be analyzed and processed further. This process is often the first step in NLP tasks.
Masking is commonly used in pre-training large language models as it helps the models learn to understand the context and meaning of words in a sentence. In this technique, a certain number of words or tokens are replaced with a special token, usually denoted as "[MASK]". The goal is to predict the original word/token that was replaced with the mask token based on the surrounding words in the sentence or context.
Segmentation is an important step in NLP tasks. It helps to break down a text into meaningful units for analysis and processing. segments refer to contiguous spans of text or sequences of tokens within a longer piece of text, which are typically defined based on certain boundaries or rules.
X_train_pooled_output, X_test_pooled_output contains predicted BERT model output from tokens, Masks, segments.

## Training a NN with 768 features
<p> When training a neural network with 768 features from BERT, it is important to ensure that your model has enough capacity to capture the complex relationships between the input features and the target labels. This may involve increasing the number of fully connected layers, increasing the number of neurons in each layer, or using a more complex architecture.
It employs 12 attention heads and 12 hidden layers, or Transformer blocks, with a hidden size of H = 768. See the BERT collection for other model sizes. The data we pass between the models is a vector of size 768. We can think of this vector as an embedding for the sentence that we can use for classification.
Creating NN with dense layers and dropouts without overfitting and underfitting.
compile the model with binary_crossentropy and adam optimizer. Created custom callback-for-keras os accuracy and target AUC
Using EarlyStopping, TensorBoard, ReduceLROnPlateau Callbacks to log all metrics and Losses. Fitting model with 100 epochs 32 bach size.
After training model we get loss: 0.2118 - val_loss: 0.1928 -AUC: 0.9487 - lr: 1.0000e-06. Logs of tensorboard represents auc and loss
The AUC graph is increasing with a rough curve and got 94% of accuracy. Loss graph is decreasing with a rough curve.

## Creating a Data pipeline for BERT Model
Subsequent to fitting the model we make an information pipeline to systematize and mechanize the work process.
Use Regular Expressions to initialize the length of the sequence, the tokenizer, and the bert model to clean strings. Map the tokens to pad the sequences.
Initialize segments in the shape of the tokens. Add [CLS] and [SEP] out the sequences.
Converts a sequence of tokens into a single id. Making pipeline capability to call bertvectorizer to foresee the class of given information as length of grouping, tokenizer and bert_model.
Predict the output of on test.csv using the previously trained neural network model.The count of data points categorized as 1 or 0 can be obtained by returning the occurrences of class labels from the function.

## Reference
1. Workflow of BERT https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
2. Illustration of BERT https://jalammar.github.io/illustrated-bert/
3. Dataset from kaggle https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
4. www.appliedaicourse.com

