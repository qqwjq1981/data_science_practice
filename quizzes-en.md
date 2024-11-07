# Data Science Interview Questions & Exercises

[Chapter 1 Introduction to Data Science](#Chapter-1-Introduction-to-Data-Science)

[Chapter 2 Statistical Foundations](#Chapter-2-Statistical-Foundations)

[Chapter 3 Machine Learning Essentials](#Chapter-3-Machine-Learning-Essentials)

[Chapter 4 Neural Networks and Deep Learning](#Chapter-4-Neural-Networks-and-Deep-Learning)

[Chapter 5 Data Science Workflow](#Chapter-5-Data-Science-Workflow)

[Chapter 6 Data Storage and Computation](#Chapter-6-Data-Storage-and-Computation)

[Chapter 7 Data Science Technology Stack](#Chapter-7-Data-Science-Technology-Stack)

[Chapter 8 Product Analytics](#Chapter-8-Product-Analytics)

[Chapter 9 Metrics](#Chapter-9-Metrics)

[Chapter 10 A/B Testing](#Chapter-10-A--B-Testing)

[Chapter 11 Models in Search, Recommendation and Advertising](#Chapter-11-Models-in-Search--Recommendation-and-Advertising)

[Chapter 12 Recommender Systems](#Chapter-12-Recommender-Systems)

[Chapter 13 Computational Advertising](#Chapter-13-Computational-Advertising)

[Chapter 14 Search](#Chapter-14-Search)

[Chapter 15 Natural Language Models](#Chapter-15-Natural-Language-Models)

[Chapter 16 Introduction to Large Language Models](#Chapter-16-Introduction-to-Large-Language-Models)

## Chapter 1 Introduction to Data Science

#### What is the difference between the Data Scientist and Machine Learning Engineer role?
> A Data Scientist primarily focuses on translating business challenges into data-driven problems, propose and prototype solutions. A Machine Learning Engineer takes these insights and turns them into concrete algorithms and models, deploy and maintain the models in a production environment.
> 
> | Attributes  | Data Scientist                   | Machine Learning Engineer         
> |--------------|----------------------------------|----------------------------|
> | Focus        | Frame business problems as data science problems, gain insights from data | Develop, optimize, deploy and maintain models in production |
> | Skillset       | Statistics, machine learning, data visualization, communication | Machine learning, algorithm design and optimization, MLOps |
> | Responsibility     | From business problem to data-driven solution | From algorithm design to system implementation |

#### Please name a few data science applications in areas such as healthcare, finance, e-commerce, and marketing.
> Healthcare: 
> * Disease diagnosis: Analyze patients' medical data (such as medical history, medical imaging, genetic information) to diagnose disease in early stage. 
> * AI-driven drug development: Discover potential drug targets to accelerate the drug development process and reduce development cost.
>     
>Finance: 
> * Fraud detection: Identify fraudulent transactions and prevent financial fraud.
> * Risk assessment: Evaluate credit risk using customer credit history, transaction records, and other data to provide decision support for loan approvals.
> * Portfolio optimization: Construct optimal investment portfolio based on market data and client risk preferences. 
> * Quantitative trading: Analyze market data to automatically carry out trading strategies.
>
>E-commerce: 
> * Recommendation system: Recommend products of interest to users based on their historical purchase records, browsing behavior, and demographic data. 
> * Demand forecasting: Predict future product demand to optimize inventory. 
> * Customer segmentation: Segment customers based on different characteristics to achieve precise marketing.
> * Price optimization: Dynamically adjust product prices based on market competition and product attributes.
> 
> Marketing: 
> * Customer Relationship Management: Understand customer needs, and improve customer satisfaction and loyalty. 
> * Advertising: Optimize advertising strategies to improve conversion or ROI.
> * Measurement: Measure the ROI of marketing activities to provide a basis for subsequent marketing decisions.

#### Provide 2-3 examples of outstanding data scientists around you, highlighting their exceptional qualities.

#### Considering your professional background and work experience, think about your career plan for the next 3-5 years. If you wish to transition to a data scientist or algorithm engineer, how should you identify and address gaps in your skills?

#### What technical skills and knowledge are essential for transitioning from small-scale data analysis to large-scale data science projects?

>* **Data Storage:** Distributed storage systems (e.g., Hadoop HDFS, AWS S3, Azure Blob Storage, NoSQL databases) and data lakes.
>* **Computation:** Big data processing frameworks (e.g., MapReduce, Spark) and parallel computing techniques.
>* **System Implementation:** Data pipelines, monitoring tools, and containerization technologies.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 2 Statistical Foundations

#### Simulate repeated coin flips until the first head appears. Determine the probability distribution of the total number of flips required.

> The distribution of the total number of tosses follows a geometric distribution, with probability mass function $P(X=k) = (1/2)^k$.

#### Monty Hall problem: The Monty Hall problem is a classic probability puzzle. Suppose there are three doors, behind one of which is a car, and behind the other two are goats. A contestant randomly selects a door. After the contestant’s choice, the host, who knows what’s behind each door, opens another door, revealing a goat. Should the contestant stick with their original choice, or switch to the remaining unopened door?

>  If the contestant initially selects a door randomly, the probability of choosing the car is $1/3$. Therefore, if they stick with their initial choice, the probability of winning remains $1/3$. However, if the contestant initially selects a goat (which has a $2/3$ probability), the host will be forced to reveal the other goat. In this case, switching to the remaining unopened door guarantees a win. Thus, the optimal strategy is to **always switch doors**.。

#### Given an unfair coin with a probability of landing heads up, $p$, how can we simulate a fair coin flip?

> A common approach is to flip the coin twice. If the two flips result in the same outcome (both heads or both tails), discard the result and flip again. If the two flips result in different outcomes (one head and one tail), we can use the first flip as the outcome of the fair coin flip. The probability of getting a (Head, Tail) or a (Tail, Head) in this scenario is 1/2.

#### Please use a real-world example to explain false positives and false negatives. 

> For instance, consider a new diagnostic test for a disease.
>* **False Positive**: A healthy individual tests positive for the disease, leading to unnecessary concern and potential further testing.
>* **False Negative**: A person with the disease tests negative, delaying diagnosis and potentially hindering timely treatment.

#### What are some common sampling techniques used to select a subset from a finite population? Please provide up to 5 examples.

> Commonly used sampling techniques include: 
>  - Sampling with replacement.
>  - Sampling without replacement.
>  - Stratified sampling.
>  - Multi-stage sampling. 
>  - Systematic sampling.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 3 Machine Learning Essentials

#### Compare and contrast the strengths and weaknesses of Linear Regression (LR), Random Forest (RF), and Gradient Boosting Decision Trees (GBDT) algorithms. Additionally, discuss the strategies for implementing these algorithms in a distributed computing environment.

> A detailed comparison of the advantages and disadvantages of these algorithms can be found in Table 3-3 of this book. Implementing these algorithms in distributed computing environment:
> - Random Forest can be easily parallelized by constructing individual decision trees concurrently. 
> - While Gradient Boosting Decision Trees are inherently sequential, feature computation and tree construction for each iteration can be parallelized in a distributed setting. 
> - Linear Regression can be parallelized using parameter server and asynchronous update mechanisms.

#### For a binary classification problem, consider randomly selecting one positive and one negative sample. The Area Under the Curve (AUC) metric can be interpreted as the probability that the model assigns a higher score to the positive sample than to the negative sample. Please provide a formal derivation of this relationship.

Let $F_S​(s)$ and $F_T​(t)$ denote the cumulative distribution functions of the predicted scores for the positive and negative classes, respectively. The probability that a randomly drawn positive sample, $S$, has a higher score than a randomly drawn negative sample, $T$, is given by $P(S>T)$. By applying probability integral transformation and recognizing the geometric interpretation of AUC is the area under the ROC curve, we can formally derive this relationship with some integral tricks.

#### What is the difference between XGBoost and GBDT algorithms?

> | Attributes | GBDT | XGBoost |
> |---|---|---| 
> | Loss Function | Primarily uses squared error loss or exponential loss | Supports custom loss functions and incorporates second-order derivatives | 
> | Regularization | No explicit regularization terms | Introduces L1 and L2 regularization terms to prevent overfitting | 
> | Optimization Algorithm | Based on gradient descent | Based on second-order Taylor expansion for more accurate approximation of objective function | 
> | Missing data | No explicit mechanism for handling missing data | Built-in strategies for handling missing data | 
> | Parallelization | Supports parallel computing, but relatively inefficient | Supports parallel computing and optimizes efficiency | 
> | System Optimization | Relatively few optimizations | Optimizations for cache, column sampling etc. |

#### What approaches can be used to combat overfitting and underfitting?

> - **Overfitting**: 
>     - Ensemble learning (Bagging, Boosting).
>     - L1 and L2 regularization.
>     - Regularization techniques in neural networks (dropout and early stopping). 
>  - **Underfitting**: 
>      - Feature crossing. 
>      - More complex models and search for new features.

#### List common distance-based clustering and density-based clustering algorithms.  Please provide up to 5 examples.

> Common distance-based clustering algorithms include K-Means and hierarchical clustering. Density-based clustering algorithms include DBSCAN, HDBSCAN, etc.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 4 Neural Networks and Deep Learning

#### What are some commonly used activation functions, and how do they compare in terms of computational complexity, sparsity, and gradient behavior?

> * **Sigmoid function**: maps input values to a range between 0 and 1, representing the probability of a positive class. While useful for binary classification, it may suffer from vanishing gradients with extreme input values.
> * **Tanh function**: transforms input values to a range between -1 and 1. Similar to Sigmoid but with a wider range, which can sometimes improve performance on data with greater variation.
>* **ReLU function**: sets negative inputs to 0 and retains positive values. ReLU is computationally simple and widely used, as it avoids the vanishing gradient issue, enhancing model performance in many applications.
> * **Leaky ReLU function**: modifies ReLU by introducing a small slope for negative values, and mitigates the “dying ReLU” problem. Leaky ReLU maintains a broader activation range, contributing to model stability.

#### What are the common regularization methods in deep learning, and how do layer normalization and batch normalization differ?

> Common regularization methods in deep learning include dropout and early stopping. Comparison between layer norm and batch norm:
> - **Layer normalization** normalizes across the feature dimensions within a single sample, making it independent of batch size.
> - **Batch normalization** normalizes across feature dimensions within a batch, which can accelerate training when input size is fixed. 
>
> For variable input sequences or when batch-size independence is required, layer normalization is preferred. Batch normalization is generally effective when input size is fixed and speed in training is a priority.

#### What are the application scenarios for one-to-one, one-to-many, and many-to-many configurations in the input and output layers of a Recurrent Neural Network?

> Recurrent Neural Networks are used in **one-to-one** scenarios like image classification, **one-to-many** scenarios such as image-to-text conversion, and **many-to-many** scenarios like text translation.

#### Why do transformers handle massive datasets better than recurrent neural networks and avoid gradient explosion?

> Transformers outperform RNNs due to their strong parallel processing capabilities, effective long-range dependency modeling, and resilience to gradient vanishing or explosion when handling large datasets. Layer normalization in transformers helps mitigate gradient explosion.

#### In which tasks have transformers excelled? What are the future trends?

> Transformers have demonstrated exceptional sequential modeling abilities, achieving remarkable results in fields such as natural language processing, computer vision, and speech recognition. Future trends include enhancing multimodal capabilities, scaling models further, reducing computational and inference costs, and integrating with technologies like reinforcement learning and knowledge graphs.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 5 Data Science Workflow

#### How should one choose between manual and automated feature engineering? In which scenarios is each approach preferable?

> The choice depends primarily on **domain knowledge** and **feature quantity**. Manual feature engineering is generally preferred when domain knowledge is rich, there is a deep understanding of the data, and the number of features is manageable. In contrast, automated feature engineering is more suitable when domain knowledge is limited, the dataset is large, manually designing features is costly, and model interpretability requirements are lower.

#### How can continuous features be bucketed based on data distribution, and what are the pros and cons of distribution-based bucketing?

> A common approach is to bucket based on data quantiles. This method ensures uniform sample sizes across buckets but results in non-integer boundaries and may be affected by distribution drift.

#### What data collection methods have you encountered or used in past work?

> Methods include recording transaction data from terminal devices, scraping web data from the internet, log user behaviors, and track sensor data from vehicles.

#### How are missing values commonly handled?

> Common methods include modeling the missing data mechanism, treating missingness as a binary variable, and employing imputation techniques like mean and regression imputation.

#### What are common methods for detecting outliers?

> Outlier detection methods are generally categorized as statistical-based and density-based.
>
> **Statistical-based methods** include:
>- **The 3$\sigma$ principle:** Assumes a normal distribution, with data points beyond the mean ± 3 standard deviations considered outliers.
>-  **Box plot method:** Identifies outliers using the upper and lower quartiles and 1.5 times the interquartile range.
>- **Z-score method:** Standardizes data and flags points as outliers if their Z-scores exceed a specified threshold.
>
>**Density-based methods** include:
> - **DBSCAN clustering:** Classifies points as core, border, or noise, with noise points identified as potential outliers.
> - **LOF (Local Outlier Factor):** Measures outlyingness by comparing the local density of each point to that of its neighbors.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 6 Data Storage and Computation

#### Design a database system for a convenience store that tracks transaction time, user, itemized products (quantity and price), and allows for product returns.

> Requirements:
> -   Track transaction time, user ID, and multiple products per order.
> -   Record product name, quantity, and unit price for each item.
>-   Accommodate product returns.
>
> Table Design:
> -  Order Table (`orders`)
> 
> | Column Name    | Type        | Description                                                          |
> |----------------|-------------|----------------------------------------------------------------------|
> | `order_id`     | Primary Key | Order ID                                                             |
> | `customer_id`  | Foreign Key | Customer ID                                                          |
> | `order_time`   | Timestamp   | Order creation time                                                  |
>| `total_amount` | Decimal     | Total order amount                                                   |
> | `status`       | Enum        | Order status (e.g., Pending Payment, Paid, Completed, Canceled)      |
>
> - Order Details Table (`order_items`)
>
>| Column Name   | Type        | Description                                      |
>|---------------|-------------|--------------------------------------------------|
>| `item_id`     | Primary Key | Order detail ID                                  |
>| `order_id`    | Foreign Key | Associated order ID                              |
>| `product_id`  | Foreign Key | Product ID                                       |
>| `quantity`    | Integer     | Product quantity                                 |
>| `unit_price`  | Decimal     | Unit price of the product                        |
>
> - Product Table (`products`)
>
> | Column Name    | Type        | Description               |
> |----------------|-------------|---------------------------|
> | `product_id`   | Primary Key | Product ID               |
> | `product_name` | Text        | Product name             |
> | `category`     | Text        | Product category         |
> | `price`        | Decimal     | Product price            |
>
> - Returns Record Table (`returns`)
>
> | Column Name       | Type        | Description                  |
> |-------------------|-------------|------------------------------|
> | `return_id`       | Primary Key | Return record ID             |
> | `order_item_id`   | Foreign Key | Associated order detail ID   |
> | `return_quantity` | Integer     | Return quantity              |
> | `return_time`     | Timestamp   | Return time                  |

#### How can we calculate the median price from a stream of transaction prices?

> We can approximate the median price by bucketing prices and tracking the frequency of each bucket in the data stream.

#### Why use a vector database when vector search packages exist?

> Vector search packages are optimized for small-scale, single-machine vector processing without persistence. Vector databases excel in large-scale, distributed environments, offering robust storage, retrieval, and advanced features. The design choice depends on factors like data volume, performance needs, and required functionality. For small-scale applications with modest performance demands, a vector search package may suffice. However, for large-scale, high-performance scenarios, a vector database is the preferred solution.

#### What are the key differences between data warehouses and data lakes?

> Refer to Table 6-7 for a comparison of data warehouses and data lakes.

#### Given a dataset of student attendance records (date, user ID, and attendance status), identify students with more than 3 consecutive absences.

> The core challenge is identifying continuous absence segments within the attendance data. This is a common **"Gaps and Islands"** problem in SQL. The solution involves sorting absence dates for each student and using a difference operation to group consecutive absences.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)
# Chapter 7 Data Science Technology Stack

#### How can an XGBoost model, trained in Python, be deployed to a production environment?

> -   **Model Serialization:** Save the trained model using XGBoost's built-in methods, Pickle, or Joblib.
> -   **Model Loading:** Load the serialized model in the production environment using the appropriate library.
> -   **Deployment Platform:** Choose a suitable deployment platform (local or cloud-based).
> -   **Model Serving:** Create a service layer (e.g., REST API, gRPC) to expose the model for predictions.

#### Compare and contrast common hyperparameter tuning techniques: Greedy Search, Grid Search, and Bayesian Optimization.

> | Method | Advantages | Disadvantages | Best Suited For | 
> |---|---|---|---| 
> | Greedy Search | Simple, efficient | Prone to local optima | Initial exploration, few parameters |
> | Grid Search | Reliable, global optimum | Computationally expensive| Small search space, abundant resources | 
> | Bayesian Optimization | Efficient, finds good solutions | Complex implementation | Large search space, limited resources |

#### How can we evaluate a click-through rate (CTR) prediction model, both offline and in real-time? What challenges might arise during this evaluation?

> Offline Evaluation:
> -   **Metric:** Cross-entropy is a common choice for evaluating CTR prediction models.
>-   **Offline evaluation:** A held-out test set is used to compare predicted CTRs with actual click outcomes.
>- **Real-time Evaluation:** Compute cross-entropy in real-time using online traffic.
>
> Challenges:
>    -   **Selection Bias:** Online traffic may be skewed towards high-potential clicks.
>    -   **Click Delay:** Time delays between exposure and click can impact metric calculation.
>    -   **Sample Leakage:** Ensuring test data is independent of training data.
>
> To mitigate these challenges, consider techniques like:
>-   **Random Traffic Evaluation:** Evaluating model performance on randomly selected traffic.
>-   **Time-Decayed Metrics:** Adjusting metric calculations to account for click delays.
>-   **Strict Data Partitioning:** Preventing test data leakage into training.

#### Outline the offline training and online deployment processes for a comment quality scoring model, along with potential technology choices.

> Offline Training:
>-  **Data Preparation:** Collect, clean, and label comment data.
>-  **Model Selection and Architecture:** Choose a suitable model architecture (e.g., TextCNN, LSTM, BERT) and define the loss function and optimizer.
>-  **Model Training:** Train the model using a deep learning framework like PyTorch and evaluate its performance on a validation set.
>
> Online Deployment:
>- **Model Serialization:** Save the trained model in a suitable format (e.g., PyTorch's `.pt` format).
>- **Deployment Platform:** Choose a deployment platform (e.g., TorchServe, TensorFlow Serving, cloud-based platforms, containerization) to host the model.
> - **Model Serving:** Create a service layer (e.g., REST API, gRPC) to expose the model for prediction requests.
> 
> Technology Stack:
> * PyTorch: Deep learning framework
> * NLTK/spaCy：NLP
> * Flask/FastAPI：Web framework
> * Docker: Containerization technology
> * Kubernetes: Container orchestration platform
> * Cloud Service Platforms: AWS SageMaker, Google Cloud AI Platform, etc.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

# Chapter 8 Product Analytics

#### How can the overall average income decrease while the average income of every ethnic group increases?

> This is a classic example of Simpson's Paradox, where the overall trend contradicts the trends within subgroups due to a confounding variable, in this case, changes in the demographic composition of the US population.

#### What is selection bias, and how can it be mitigated in research?

> Selection bias occurs when the sample used for a study is not representative of the population of interest. This can lead to misleading conclusions. Common types of selection bias include:
> -   **Survivor Bias:** Focusing on surviving cases, ignoring those that didn't.
>-   **Response Bias:** Non-response or biased responses from certain groups.
>-   **Truncation Bias:** Excluding certain groups based on a specific criterion.
>-   **Confirmation Bias:** Seeking evidence that confirms existing beliefs.
>
>To mitigate selection bias:
>-   **Random Sampling:** Ensure a representative sample from the population.
> -   **Random Assignment:** Randomly assign subjects to treatment and control groups.
> -   **Control Groups:** Use appropriate control groups for comparison.
> -   **Statistical Techniques:** Employ techniques like post-stratification, regression analysis, or propensity score matching to adjust for bias.

#### Describe the user registration funnel for a product you frequently use. Identify potential optimization areas and propose data-driven strategies to improve the funnel.

> Refer to Figure 8-13 for a visual representation of a ride-hailing driver's registration funnel.

#### How can we analyze the supply-demand dynamics of user-generated content (e.g., short videos, images) on social platforms?

>Example: Short Video Content.
> **Supply-Side Analysis:**
> - **Upload Volume:** Analyze the daily volume of uploaded short videos and identify trends.
> - **Topic Categorization:** Classify videos by vertical (e.g., food, travel, education) to identify popular and niche segments.
> - **Video Quality:** Assess video quality (clarity, editing, originality) and analyze the distribution of video quality levels.

> **Demand-Side Analysis:**
> - **Viewership:** Analyze daily video views and identify trends.
> - **User Engagement:** Assess user engagement through metrics like likes, comments, and shares to understand viewer preferences.
> -  **User Demographics:** Analyze user demographics (gender, age, region) to identify content preferences based on user segments.
> 
>This approach provides a foundational framework for analyzing the supply-demand dynamics of short video content on social platforms.

#### An e-commerce platform experienced an 8% year-over-year increase in GMV. Analyze the potential drivers of this growth using data-driven insights.

> To uncover the drivers of this growth, consider the following analyses:
>
> -   **User Segmentation:** Analyze GMV trends across different user segments (gender, location, new vs. returning users).
> -   **Product Segmentation:** Examine GMV performance by product category to identify top-performing categories.
> -   **Time Series Analysis:** Analyze GMV trends over time (monthly, weekly, daily) to identify seasonal or cyclical patterns.
> -   **Traffic Source Analysis:** Analyze GMV performance across different traffic sources (homepage, category pages, search results) to identify high-converting channels.
>
>By conducting these analyses, we can gain valuable insights into the factors contributing to the GMV growth and identify potential areas for further optimization.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 9 Metrics

#### How should Netflix evaluate the impact of a price reduction from $11.99 to $9.99, which led to a 6% increase in subscribers?

> A key metric to consider is the impact on overall revenue. While a 6% subscriber increase is positive, it must be weighed against the revenue loss from the price reduction. Other factors, such as operating costs, potential advertising revenue, and long-term subscriber retention, should also be factored into the decision.

#### How can we measure user engagement and content quality on platforms like Toutiao?

> Here are some potential metrics to consider: 
> - User interaction metrics: Click-Through Rate (CTR), time spent, likes, comments, shares, etc.
> - Content quality metrics: Content relevance, freshness, diversity.
> - User feedback metrics: NPS from user survey, user feedback.

#### What are key business metrics for short video platforms like Tiktok?

> Key business metrics for Short Video Platforms:
> -  User Growth: Daily Active Users (DAU), Monthly Active Users (MAU), User Retention Rate.
> -  User Engagement: Video Views, Likes, Comments, Shares.
> -  Monetization: Advertising Revenue, E-commerce Revenue, Paying User Ratio.
> -  Content Ecosystem: Content Creation Rate, Number of Creators, Creator Revenue.

#### How can we measure search result relevance?

> Measuring Search Relevance:
> -   Subjective Evaluation: define evaluation criteria, manually label search results or use LLMs, calculate metrics like Normalized Discounted Cumulative Gain (NDCG).
>
> -   User behavior metrics:
>        -   Click-Through Rate (CTR)
>        -   Long Play Rate
>        -   Bounce Rate
#### How can we balance content quality and user engagement when moderating low-quality content?

>* Suppressing low-quality content can impact user engagement and creator retention. A balanced approach requires careful consideration of platform tone, revenue goals, and creator incentives.
>
>* Define the platform's long-term vision: Should it prioritize high-quality content for user satisfaction or focus on content quantity for rapid growth?
>
> * Quantify the impact of low-quality content on user engagement by analyzing metrics like time spent and engagement rates. Conduct A/B tests to measure the effectiveness of different content moderation strategies.
>
>* Consider the broader impact of content moderation on the platform's tone, revenue, and creator ecosystem. Develop strategies to incentivize high-quality content creation while minimizing negative impacts on the community.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 10 A/B Testing

#### What are the common causes of sample ratio mismatch (SRM) in A/B testing, and how can we mitigate it?

> Sample ratio mismatch (SRM) can occur due to various factors, including:
>
> -   **Randomization Issues:** The random generator may not allocate users evenly to test and control groups.
> -   **Technical Implementation Errors:** Differences in how test variants are delivered or logged.
>-   **Data Quality Measures:** Filtering out bots or low-quality traffic can unintentionally skew the sample.
>
> To mitigate SRM and ensure accurate A/B test results:
>
> -   **Monitor SRM Ratios:** Continuously track the distribution of users across test and control groups.
> -   **Verify Randomization Effectiveness:** Ensure that users are evenly distributed across buckets.
> -   **Employ Statistical Techniques:** Use methods like propensity score matching (PSM) to control for confounding variables.

#### A/B test can be conducted at user level or device level. For new user registration, shall we use device-level or user-level?

> **For new user registration, device-level A/B testing is generally preferred.**
>This approach offers several advantages:
> -   **Reliable Identification:** Device IDs provide a consistent identifier, even for anonymous users.
> -   **Avoids Multiple Exposures:** Ensures each device sees only one variant.
> -   **Simplified Implementation:** Easier to set up and track.
>
>However, be mindful of potential limitations:
>-   **Multiple Devices per User:** A single user might use multiple devices, potentially diluting the test's impact.
>
>To mitigate this, consider using a combination of device and user-level data or advanced statistical techniques.
>
#### What is Poorman's bootstrap, and how can it be implemented?

> Poorman's bootstrap is a technique used to estimate the variance of a statistic, particularly in the context of A/B testing. It involves dividing the data into multiple buckets, calculating the statistic for each bucket, and then estimating the variance based on the variability between the bucket-level statistics.
>
> Pseudocode for Poorman's Bootstrap:
> 1.  Divide the data into `B` buckets of equal size.
>2. Calculate Statistic for Each Bucket: For each bucket `i`, calculate the statistic of interest (e.g., mean, difference in means).
>3.  Estimate Variance: Calculate the variance of the `B` bucket-level statistics, use this variance as an estimate of the true variance of the statistic.
>
>By using Poorman's bootstrap, we can obtain a more accurate estimate of the variance, leading to more reliable statistical inferences.

#### How can we reduce the variability of experimental metrics?

> Common techniques to stabilize experimental metrics include:
> -   **Data Transformation:** Applying transformations like log or square root to normalize the distribution.
> -   **Truncation:** Limiting the range of the metric to reduce the impact of extreme values.
> -   **Outlier Removal:** Identifying and removing outliers that can distort the results.
> -   **Statistical methods:** Using techniques like CUPED or stratified sampling to control for confounding factors.

#### To what extent has your company integrated experimentation into its decision-making process (crawling, walking, running, or flying)?

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 11 Models in Search, Recommendation and Advertising

#### How do search and recommendation systems differ in terms of user intent, business objectives, user profiling, and personalization?

> Refer to Table 11-1 for a detailed comparison.

#### What is the TF-IDF formula, and how does it work?

> Refer to Chapter 11.2.1 for a detailed definition.

#### How does the PageRank algorithm work?

> The PageRank algorithm iteratively assigns a rank to each webpage based on the number and quality of incoming links. Here's a simplified overview of the process:

> 1.  **Initialization:** Assign an equal initial PageRank value to each webpage.
> 2.  **Link Analysis:** Calculate the out-degree of each webpage (the number of outgoing links).
> 3.  **Damping Factor:** Set a damping factor (usually 0.85) to account for random surfing behavior.
> 4.  **Iterative Update:** For each webpage:
>       a. Calculate the sum of PageRank values from incoming links, weighted by the source page's out-degree.
>       b. Multiply this sum by the damping factor.
>       c. Add a damping factor multiplied by a constant (usually 1/N, where N is the total number of webpages).
> 5.  **Convergence:** Repeat the iterative update process until the PageRank values stabilize or a maximum number of iterations is reached.
>
> The mathematical representation of the PageRank update formula is:
>$$R = dMR + \frac{1-d}{n} 1 $$
>
>where $R$ is the $n$-dimensional PageRank value vector, $M$ is the state transition matrix, and $1$ represents an n-dimensional vector with all elements equal to 1.

#### How does a two-tower neural network model work in both offline training and online serving?

> **Offline Training:**
> -  **Feature Engineering:** Extract and preprocess user and item features.
> - **Model Architecture:** Construct separate neural networks for users and items.
> - **Negative Sampling:** Generate negative samples to train the model to distinguish between relevant and irrelevant items.
> - **Model Training:** Train the model to learn embeddings that capture user preferences and item attributes.
>
> **Online Serving:**
> -  **User Feature Extraction:** Extract features for the active user.
> - **Similarity Calculation:** Calculate the similarity between the user's embedding and the embeddings of all items.
> - **Ranking:** Rank items based on similarity scores and return the top N recommendations.

#### How does the FTRL algorithm differ from SGD in terms of regularization, learning rate adaptation, historical information utilization, and application scenarios?

> | Feature | SGD | FTRL |
> |---|---|---| 
> | Regularization | L1/L2 | Adaptive L1+L2 | 
> | Learning Rate | Fixed or Decaying | Adaptive | 
> | Historical Information | Uses less | Uses fully | 
> | Use Cases | General | Sparse Data |

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 12: Recommender Systems

#### How can we evaluate the performance of a recommendation retrieval model?

> To evaluate a recommendation retrieval model, we can consider both individual and comparative metrics:
>
> **Evaluation of a single retrieval path**:
>    -   Top-K Accuracy
>     -   Retrieval Rate
>     -   F1-Score
>     -   Hit Rate
>
> **Evaluation in the presence of other retrieval paths**:
> >
> - **Retrieval Overlap:** Measures the extent to which a new retrieval source complements existing ones.
>  - **Ranking Consistency:** Assesses how well the retrieval items aligns with the ranking model.

#### How do the retrieval, ranking, and re-ranking modules of a recommendation system work?

> Refer to Section 12.2 for more details.

#### How can we select effective ranking metrics for feeds recommendations to improve user retention?

> To optimize for user retention, we can use a regression model with user retention as the target variable and user engagement metrics (e.g., time spent, likes, comments) as features. The resulting model weights can inform relative importance of these engagement metrics. By prioritizing content that drives these engagement metrics, we can increase user retention.

#### How can we enhance the diversity of recommendation results?

> We can enhance recommendation diversity through:
> 
> -  **Post-Processing:** Adjusting the ranking results to disperse similar items.
> -  **List-Level Reordering:** Using techniques like Determinantal Point Processes (DPP) to optimize the overall diversity of the recommendation list.

#### What are the key components of a user profile, and how can we build a user understanding platform?

> Key Dimensions of User Profiles:
>
> -   **Demographic Information:** Age, gender, location, etc.
> -   **Interests:** Topics, genres, brands, etc.
> -   **Behavioral Data:** Browsing history, purchase history, click-through rates, etc.
> 
> Building a User Profiling Platform:
>
>-  **Tagging System:** Develop a system to categorize and tag user content and behavior.
>-  **User Classifiers:** Train models to identify user interests and preferences based on their historical data.
> -  **User Interface:** Create a user-friendly interface to visualize user profiles and insights.
> - **API Integration:** Develop APIs to expose user profile data to other systems.
> - **Application Scenarios:** Identify use cases for personalized recommendations, targeted marketing, and user segmentation.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

# Chapter 13 Computational Advertising

#### How do brand advertising and performance advertising differ in terms of goals, duration, optimization strategies, and revenue measurement?

> Please refer to Table 13-1 for a detailed comparison.

#### How can we diagnose the reasons for a lack of ad impressions and budget spending?

> To diagnose the issue, consider the following:
>
> -   **Targeting:** Ensure the target audience is defined accurately and not overly restrictive.
> -   **Bidding:** Verify that the bidding strategy is competitive and the bid amount is sufficient.
> -   **Ad Creative:** Assess the quality and relevance of the ad creative.
>-   **Account Status:** Check for any account suspensions, restrictions, or daily budget limitations.

#### How does social media advertising on platforms like Facebook work, from ad creation to conversion tracking?

> Social Media Advertising Process:
> 1.  **Ad Creation:**
    -   Define advertising objectives (e.g., brand awareness, website traffic, lead generation).
    -   Create ad groups and design ad creatives (e.g., images, videos, text).
> 2. **Budget and Bidding:**
    -   Set a daily budget.
    -   Choose a bidding strategy (e.g., cost-per-click, cost-per-impression, automated bidding).
> 3. **Ad Targeting:**
    -   Define the target audience based on demographics, interests, and behaviors.
> 4. **Ad Delivery:**
    -   The platform's algorithm displays ads to the target audience.
> 5. **Ad Conversion:**
    -   Track user actions after ad clicks (e.g., website visits, purchases, sign-ups).

#### What are common techniques for pacing advertising budgets?

> Two common budget pacing techniques are:
> - **Bid adjustments**: enhancing ad spending by modifying bids;
> - **Probabilistic throttling**: control budget spend by changing the probability of auction
   participation.

#### How does budget-aware experiments address the challenge of budget allocation in A/B testing?

> Budget-aware experimentation addresses the challenge of resource allocation in A/B testing. By isolating budgets for different experiments, we can avoid interference and ensure accurate performance evaluation.
>
> Here's how to implement a budget-isolated A/B test:
> 
> 1.  **Allocate Budget:** Assign a specific budget to each experiment.
> 2. **Independent Pacing:** Implement independent pacing mechanisms to control spending for each experiment.
> 3. **Statistical Analysis:** Use statistical techniques to analyze the results, accounting for budget constraints.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 14 Search

#### How is an inverted index constructed?

> Refer to Figure 14-3 for a visual representation of the inverted index construction process: tokenization, text processing, and indexing.

#### How is the Normalized Discounted Cumulative Gain (NDCG) metric calculated, and what does it measure?

> For a detailed explanation and example calculation of NDCG, please refer to Section 14.3.2.
> 
#### How can we assess the relevance and authority of search results?

> **Relevance:**
>
> -   **Subjective Evaluation:** Human experts assess result relevance based on predefined guidelines.
> -   **User behavior:** Analyze user behavior metrics like click-through rate (CTR), dwell time, and bounce rate.
>
> **Authority:**
>-   **Website Authority:** Evaluate the reputation and trustworthiness of the website.
>-   **Author Authority:** Assess the expertise and credibility of the author.
>-   **Author-Category Authority:** Consider the author's specific knowledge and experience in the relevant field.
>
#### How can search terms be categorized?

> Search terms can be categorized by:
>
> -   **Generality:** Broad-intent or precise intent
> -   **Timeliness:** Timely or non-timely
> -   **User Intent:** Navigational, informational, or transactional

#### Compare and contrast subjective and objective evaluation methods for search.

> | Area | Subjective Evaluation | User Behavior | 
> |---|---|---| 
> | Granularity | Low granularity | High granularity | 
> | Accuracy | Relies on human judgment | Affected by data noise | 
> | Applicability | End-to-end and module-level | Primarily end-to-end | 
> | Cost | High (human labor) | High (development and testing) |

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 15 Natural Language Processing

#### How do negative sampling and hierarchical Softmax work in Word2Vec?

> For a detailed explanation, please refer to Section 15.3.2.
> 
#### How can we derive sentence embeddings from word embeddings?

> Common approaches to obtain sentence embeddings include:
> -   **Averaging Word Embeddings:** Calculate the average of the word embeddings in the sentence.
> -   **Weighted Averaging:** Assign weights to words based on their importance.
> -   **Sentence Embeddings Models:** Utilize specialized models like Sentence Transformers or Sentence-BERT.
> 
#### What is a token in Natural Language Processing, and what are common tokenization techniques?

> In Natural Language Processing, a token is the smallest unit of text. It can be a word, a subword (e.g., "un-", "friend"), or a punctuation mark. Common tokenization techniques include:
>
> -   **Word-Level Tokenization:** Splits text into words.
> -   **Subword Tokenization:** Divides words into smaller units (subwords).
> -   **Character-Level Tokenization:** Breaks text into individual characters.

#### How do LSTM and transformer models compare in Natural Language Processing?

> Both LSTM and transformer models are powerful tools for processing sequential data. However, transformers excel at capturing long-range dependencies and can be parallelized more efficiently, making them a popular choice for many natural language processing tasks.

#### What are the key drivers and applications of Natural Language Processing (NLP)?

> For a deeper dive into the history and applications of NLP, please refer to Chapters 15.1 and 15.2.

[\[↑\] Back to top](#Data-Science-Interview-Questions--Exercises)

## Chapter 16 Large Language Models

#### What are the key factors that determine the scale of a Large Language Model (LLM)?

> The scale of an LLM is primarily influenced by three factors:
>
> -  **Model Size:** The number of parameters in the model.
> - **Data Scale:** The amount of training data used.
> - **Computational Resources:** The computational power required for training.

#### What are the types of hallucinations in LLMs, and how can we mitigate them?

> There are two main types of hallucinations in LLMs:
>
> -  **Intrinsic Hallucinations:** The model generates text that contradicts the provided input.
>- **Extrinsic Hallucinations:** The model generates text that is factually incorrect or unsupported by the input.
>
> To reduce hallucinations, techniques like decoding strategy optimization and retrieval-augmented generation can be employed.

#### How does Retrieval-Augmented Generation (RAG) work?

> The RAG process involves:
>
> 1. **Index Construction:** Preprocessing text data, segmenting it into tokens, and creating embeddings for efficient search.
> 2. **Retrieval:** Retrieving relevant information from a knowledge base using techniques like keyword search, semantic search, or knowledge graph-based retrieval.
> 3. **Generation:** Using a language model to generate text based on the retrieved information and the prompt.

#### What are the data challenges associated with large language models?

> LLMs face significant data challenges, including:
>
> -   **Data Quality:** Ensuring data accuracy, consistency, and relevance.
> -   **Data Quantity:** Acquiring and processing massive amounts of data.
> -   **Data Privacy:** Protecting sensitive information and complying with data privacy regulations.
> -   **Data Bias:** Mitigating biases in the training data to avoid unfair or harmful outputs.

#### How is an LLM fine-tuned for a specific task?

> Fine-tuning an LLM involves the following steps:
> 1. **Data Preparation:** Collect and curate a dataset specific to the target task.
> 2. **Model Selection:** Choose a pre-trained LLM as a starting point.
> 3. **Model Adaptation:** Adjust the model's parameters to fit the specific task.
> 4. **Training:** Train the model on the task-specific dataset.
>  5. **Evaluation:** Assess the model's performance on a validation set.
>   6. **Deployment:** Deploy the fine-tuned model for real-world applications.