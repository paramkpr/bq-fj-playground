![](https://fischerjordan.com/wp-content/uploads/2019/03/FJ-logo-horizontal-blue-medium-b.jpg)

# FJ x BQ: Tech Background & Research
*Param Kapur, 09/21/23*

---
## Introduction

The endeavor to manage and derive insights from a vast volume of unstructured data, such as SEC filings, necessitates a technological framework that is adept at handling, searching, and synthesizing information efficiently. Central to this framework are text embeddings, which convert textual data into vector representations, facilitating semantic similarity-based search and retrieval operations. The Massive Text Embedding Benchmark (MTEB) provides a robust measure to evaluate and compare various text embedding models across diverse tasks, ensuring an informed selection of models. Vector databases serve as the repository and retrieval system for the vectorized data, enabling efficient similarity search operations. Additionally, employing a modular architecture for integrating embeddings, vector databases, and Language Models (LLMs) fosters a conducive environment for information lookup and synthesis, while also allowing for the systematic evaluation and optimization of different tool combinations. This introductory overview outlines the primary components and considerations essential for developing a proficient system capable of intelligent information management and synthesis.

---

## Background

We'll cover the following topics one-by-one: Embeddings, MTEB, Vector Databases, and Implementation


---
### 1. Embeddings

#### a. What are embeddings?

Embeddings are a type of representation for text data where words or phrases are mapped to vectors of real numbers in a continuous vector space. The key idea behind embeddings is to have fixed-size vectors for each word, regardless of the vocabulary size, which can capture semantic relationships between words. 

###### Illustration:
> There can be many different types of similarities between words, for example, word big is similar to bigger in the same sense that small is similar to smaller. Example of another type of relationship can be word pairs big - biggest and small - smallest. We further denote two pairs of words with the same relationship as a question, as we can ask: "What is the word that is similar to small in the same sense as biggest is similar to big?"
>
> Somewhat surprisingly, these questions can be answered by performing simple algebraic operations with the vector representation of words. To find a word that is similar to small in the same sense as biggest is similar to big, we can simply compute vector ```X = vector(”biggest”) − vector(”big”) + vector(”small”)```. Then, we search in the vector space for the word closest to X measured by cosine distance, and use it as the answer to the question (we discard the input question words during this search).
>
> Finally, we found that when we train high dimensional word vectors on a large amount of data, the resulting vectors can be used to answer very subtle semantic relationships between words, such as a city and the country it belongs to, e.g. France is to Paris as Germany is to Berlin.
>

*[Efficient Estimation of Word Representations in Vector Space, Tomas Mikolov and Kai Chen and Greg Corrado and Jeffrey Dean, https://arxiv.org/abs/1301.3781]*


#### b. How do embeddings work?

Embeddings are trained using neural networks on large text corpora. Here are the steps involved in training a simple word embedding model using a method like Word2Vec:

1. **Vocabulary Creation**: Create a vocabulary from the training corpus, assigning a unique identifier to each word.
2. **Context and Target Words**: For each word in the corpus, define a context window. The words within this window are the context words, and the word itself is the target word. (This is labeling the data.)
3. **Neural Network Training**: 
    - Input Layer: The input layer contains the one-hot encoded vector of the target word.
    - Hidden Layer: The hidden layer contains the word embedding vectors. The dimensionality of this layer (number of neurons) defines the size of the word vectors.
    - Output Layer: The output layer predicts the context words using a softmax function. 

The objective is to minimize the difference between the predicted probabilities and the actual distribution of the context words. The loss function commonly used is the negative log likelihood loss.

Mathematically, the objective function \( J \) is given by:

\[
J = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)
\]

where:
- \( T \) is the total number of words in the corpus,
- \( c \) is the size of the context window,
- \( w_t \) is the target word, and
- \( w_{t+j} \) is a context word.

#### c. Why are vector embeddings and semantic similarity relevant for our use case?

Vector embeddings capture the semantic essence of the text. When dealing with a large volume of unstructured data like SEC filings, embeddings can significantly help in:

1. **Information Retrieval**: By converting text into a vector space, you can use similarity measures to retrieve documents relevant to a particular query, improving search functionality.
2. **Text Clustering**: Grouping similar documents together based on semantic similarity, aiding in data organization and retrieval.
3. **Text Summarization**: Generating concise summaries of large text corpora, enabling quicker insights.

Semantic similarity allows the system to understand the meaning behind the text, which is crucial for tasks like search and synthesis in large datasets.

#### d. What makes one embedding model better than another?

The effectiveness of an embedding model can be assessed based on:

1. **Performance**: Performance on downstream tasks like classification, clustering, or information retrieval.
2. **Training Data**: The quality and size of the training data, and how well it represents the domain of interest.
3. **Dimensionality**: Lower-dimensional embeddings are easier to work with, but there's a trade-off between dimensionality and the amount of captured information.
4. **Computational Efficiency**: The time and resources required to train and use the embeddings.
5. **Interpretability**: Some models offer more interpretable embeddings, revealing clear semantic relationships.

#### e. BONUS: How does self-training an embedding model work? And why might we choose to go down this route?

Self-training is a form of semi-supervised learning. Here’s how it works:

1. **Initial Training**: Train an embedding model on a small labeled dataset.
2. **Label Propagation**: Use the trained model to predict labels for the unlabeled data.
3. **Model Re-training**: Combine the original labeled data with the newly labeled data, and re-train the model.

This process can be iteratively repeated to improve the model's performance, especially when labeled data is scarce but unlabeled data is abundant.

Self-training can be beneficial when:
- There's a large amount of unlabeled data available.
- Labeling data is costly or time-consuming.
- You want to improve the model's performance or adapt it to a specific domain.

Great, let's delve into the Massive Text Embedding Benchmark (MTEB) based on the information available and the provided research paper.


---


### 2. MTEB (Massive Text Embedding Benchmark)

Hugging Face's MTEB is a useful tool for comparing the performance of various text-embeddings. This research paper explains how it works. *[MTEB: Massive Text Embedding Benchmark (Mar '23), Niklas Muennighoff and Nouamane Tazi and Loïc Magne and Nils Reimers, https://arxiv.org/abs/2210.07316]* The following analysis is based on it. 



#### a. What factors is the benchmark based on?

The MTEB aims to provide a comprehensive evaluation of text embedding models across a wide range of tasks and datasets. Here are the key takeaways:

- **Diversity of Tasks**: MTEB spans 8 embedding tasks, intending to cover a variety of use cases where text embeddings are applied. (Bitext Mining, Classification, Clustering, Pair Classification, Reranking, Retrieval, Semantic Textual Similarity, Summarization)
- **Extensive Dataset Coverage**: It covers a total of 58 datasets across 112 languages, aiming for a broad evaluation across different languages and data domains.
- **Model Evaluation**: Through benchmarking 33 models on MTEB, the authors aim to provide a comprehensive evaluation of text embeddings to date.
- **Evaluation Regime**: The evaluation of text embedding models in MTEB appears to be aimed at covering a breadth of applications, unlike previous evaluations that may have focused on a narrow set of tasks or datasets.


#### b. What are some examples and use cases of the top-performing pre-trained models?

The paper mentions that there isn't a particular text embedding method that dominates across all tasks, indicating that the field has yet to converge on a universal text embedding method. This suggests that different embedding models might excel in different tasks, and it's crucial to choose the appropriate model based on the specific use case.

Text embeddings power a variety of use cases from clustering, topic representation, search systems, text mining, to feature representations for downstream models.

---

### 3. Vector Databases

#### a. What are vector databases? How do they work? What makes them different from traditional databases?

- **Definition and Functionality**:
   - Vector databases are specialized databases designed to handle vector data efficiently, which is commonly generated from machine learning models. They provide the ability to perform operations like nearest neighbor search, which is essential for tasks like similarity search and clustering.

- **Working Mechanism**:
   - **Indexing**: Vector databases index high-dimensional vectors to facilitate efficient search and retrieval. Indexing strategies like KD-trees, Ball trees, or hashing techniques like Locality-Sensitive Hashing (LSH) are employed.
   - **Querying**: They allow querying based on vector similarity rather than exact matches. Queries return the most similar vectors based on a similarity or distance metric like cosine similarity or Euclidean distance.

- **Differences from Traditional Databases**:
   - **Data Type**: Traditional databases handle structured data with well-defined schemas, while vector databases are optimized for high-dimensional vector data.
   - **Query Mechanism**: Traditional databases use SQL or similar languages for exact matching queries, whereas vector databases use similarity search.
   - **Indexing**: Indexing in vector databases is tailored for high-dimensional spaces, which is significantly different from the indexing mechanisms used in traditional databases.

#### b. How should large amounts of data (like SEC filings) be seeded into a vector database?

- **Pre-processing**:
   - The text data should be pre-processed (cleaned, tokenized) and then converted into vector representations using a suitable text embedding model.

- **Batch Indexing**:
   - Once the vectors are generated, batch indexing can be used to seed large amounts of data into the vector database. This is typically more efficient than indexing each vector individually.

- **Parallel Processing**:
   - Parallel processing and distributed systems can be employed to speed up the seeding process, especially when dealing with massive datasets.

- **Incremental Updates**:
   - For continuous ingestion of new data, incremental updates to the vector database should be supported to ensure the data remains current without requiring a complete re-index.

#### c. What metrics should be used to measure vector database performance?

- **Performance Metrics**:
   - **Query Latency**: The time it takes to execute a query and retrieve results.
   - **Indexing Speed**: How fast new data can be indexed.
   - **Scalability**: How well the database handles increasing amounts of data and query load.
   - **Precision and Recall**: Accuracy of the search results in terms of relevance and completeness.

- **Other Considerations**:
   - **Ease of Use**: How user-friendly the database is, including the availability of documentation, community support, and intuitive APIs.
   - **Flexibility**: Support for different distance metrics, data types, and query mechanisms.
   - **Cost**: Total cost of ownership including licensing, hardware, and operational costs.
   - **Maintenance**: How easy it is to manage, monitor, and troubleshoot the database.


---

### 4. Implementation

#### a. Integrating a system using an embedding model, feeding into a vector database, with an LLM layer. 
[![](https://mermaid.ink/img/pako:eNpNkc1OwzAQhF9l5XP6Aj0gURIQEoifFi4Jh228TSxiO9jr_kB5d9YBCjd7NfvNjP2hWq9JzVUXcOxhtWgcwHldIiPcB5qNwbcUo3HdC8xmZ8eVfyVn3pGNdwU8kvVbHMBvILIfdz7oWMCK9gzOB4vDSVmSXFwHO8M9uGQpmFYWGdcDxSMs6mmpsmvSOpvlGIvJ8YmNYAg4C-hXAFZiD0e4qJ-pZR8gJ15jJFgSnQgXE-Haadr_rYpdWT8kCgdpeGqX5eUk_xnCW5YUcEWOAjJ932E72RWwNNYMGAwfIBKGtj9CVV-7Ta6dK8Py4LinaOKEriZ0ZcfBH-AGXZewI7jNJYSVrBXUOxVw7uKOQvaKmRL_-RsXTdezxL-s7xKPiV9UoeQlLRotP_iRfRolppYaNZejxvDaqMZ9ig4Te4nUqjmHRIVKoxZoaVA-3qr5Bod4mlbaSMef4ecXoI_C4Q?type=png)](https://mermaid.live/edit#pako:eNpNkc1OwzAQhF9l5XP6Aj0gURIQEoifFi4Jh228TSxiO9jr_kB5d9YBCjd7NfvNjP2hWq9JzVUXcOxhtWgcwHldIiPcB5qNwbcUo3HdC8xmZ8eVfyVn3pGNdwU8kvVbHMBvILIfdz7oWMCK9gzOB4vDSVmSXFwHO8M9uGQpmFYWGdcDxSMs6mmpsmvSOpvlGIvJ8YmNYAg4C-hXAFZiD0e4qJ-pZR8gJ15jJFgSnQgXE-Haadr_rYpdWT8kCgdpeGqX5eUk_xnCW5YUcEWOAjJ932E72RWwNNYMGAwfIBKGtj9CVV-7Ta6dK8Py4LinaOKEriZ0ZcfBH-AGXZewI7jNJYSVrBXUOxVw7uKOQvaKmRL_-RsXTdezxL-s7xKPiV9UoeQlLRotP_iRfRolppYaNZejxvDaqMZ9ig4Te4nUqjmHRIVKoxZoaVA-3qr5Bod4mlbaSMef4ecXoI_C4Q)


#### b. How do we measure the performance of our implementation?

1. **Efficiency Metrics**:
   - **Query Latency**: Measure the time it takes to process a query and retrieve results.
   - **Indexing Speed**: Evaluate the speed at which new data can be indexed into the vector database.

2. **Accuracy Metrics**:
   - **Precision and Recall**: Assess the relevance and completeness of the retrieved information for a set of test queries.
   - **NLP Task Performance**: Evaluate the performance on specific NLP tasks like summarization, question answering, etc., using standard metrics like BLEU, ROUGE, etc.

3. **Scalability**:
   - Assess how well the system handles increasing amounts of data and query load. 

#### c. What is a good architecture to facilitate the implementation and testing of multiple combinations of tools on a sample dataset?

![](https://showme.redstarplugin.com/d/d:Cj2v0Sfy)


---

### 5. Future Looking Questions

#### a. What features can give our application a feeling of "self-awareness" through automation?

1. **Continuous Learning**:
   - Implementing a mechanism where the system can learn from user interactions and feedback to improve its performance over time.

2. **Anomaly Detection**:
   - Enable the system to identify anomalies in the data or in its performance, and alert administrators or take corrective actions automatically.

3. **Adaptive Indexing**:
   - Allow the system to adapt its indexing strategies based on the evolving nature of the data, ensuring optimal performance as the data grows and changes.

4. **Automated Model Tuning**:
   - Incorporate automated machine learning (AutoML) techniques to continually optimize model parameters and configurations based on the performance metrics.

5. **Predictive Analysis**:
   - Utilize predictive analytics to anticipate user needs, optimize system performance, and pre-empt potential issues.

6. **Self-Healing**:
   - Implement self-healing mechanisms to automatically recover from failures or performance degradation.

7. **Context Awareness**:
   - Enable the system to be aware of the context in which it is operating, adjusting its behavior based on the current state of the data, user interactions, and external factors.

8. **Automated Reporting**:
   - Develop automated reporting and dashboards that provide insights into system performance, user behavior, and the value being delivered by the application.

9. **Real-time Monitoring and Adjustment**:
   - Establish real-time monitoring to track system pe arformance and make real-time adjustments to ensure optimal operation.

10. **Feedback Loops**:
    - Create feedback loops where user feedback and system metrics are used to continuously improve the system.

These features can contribute to creating a level of "self-awareness" in the application, where it can adapt, learn, and evolve over time to better meet the needs of its users and the goals of the organization.

The integration of self-awareness features may also necessitate a thorough review of the ethical implications, user privacy considerations, and potential biases in the system's behavior and responses.

---

### Conclusion
In conclusion, the exploration of embeddings, Massive Text Embedding Benchmark (MTEB), and vector databases presents a robust framework for handling, searching through, and synthesizing large volumes of unstructured data such as SEC filings. The utilization of suitable text embeddings transforms textual data into a format conducive for efficient similarity search and retrieval operations within vector databases. The MTEB, by providing a comprehensive benchmark across a diverse range of tasks and datasets, enables the evaluation and comparison of different embedding models, ensuring the selection of the most effective model for the specified use case. Implementing a modular architecture facilitates the integration of these components along with Language Models for information lookup and synthesis, while also allowing for the efficient evaluation of different combinations of tools on a sample dataset. Performance metrics such as query latency, indexing speed, precision, and recall are crucial for assessing the efficacy and scalability of the implementation. This technical groundwork lays a solid foundation for developing a system capable of intelligently and efficiently managing vast amounts of unstructured data, thereby significantly augmenting the information retrieval and synthesis capabilities of the application.

