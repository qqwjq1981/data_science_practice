

# Data Science Interview Questions & Practical Exercises

[Chapter 1 Introduction to Data Science](#第1章-数据科学简介)

[Chapter 2 Statistical Foundations](#第2章-统计基础)

[Chapter 3 Machine Learning](#第3章-机器学习)

[Chapter 4 Neural Networks and Deep Learning](#第4章-神经网络与深度学习)

[Chapter 5 Data Science Workflow](#第5章-数据科学的流程)

[Chapter 6 Data Storage and Processing](#第6章-数据存储和处理)

[Chapter 7 Data Science Technology Stack](#第7章-机器学习技术栈)

[Chapter 8 Product Analytics](#第8章-产品分析)

[Chapter 9 Metrics](#第9章-度量指标)

[Chapter 10 A/B Testing](#第10章-AB实验)

[Chapter 11 Models in Search, Recommendation and Advertising ](#第11章-搜索推荐广告模型)

[Chapter 12 Recommender Systems](#第12章-推荐领域的应用场景)

[Chapter 13 Computational Advertising](#第13章-广告领域的应用场景)

[Chapter 14 Search](#第14章-神经网络与深度学习)

[Chapter 15 Natural Language Models](#第15章-自然语言模型与应用场景)

[Chapter 16 Introduction to Large Language Models](#第16章-大语言模型)

## Chapter 1 Introduction to Data Science

#### What is the difference between the data scientist and machine learning engineer roles?

> A data scientist primarily focuses on translating business challenges into data-driven problems, propose and prototype solutions. A machine learning engineer takes these insights and turns them into concrete algorithms and models, deploy and maintain the models in production environment.
> | Area      | Data Scientist                   | Machine Learning Engineer         
> |--------------|----------------------------------|----------------------------|
> | Focus        | Frame business problems as modeling problems, extract insights from data | Develop, optimize, deploy and maintain models in production |
> | Skillset       | Statistics, machine learning, data visualization, communication | Machine learning, algorithm design and optimization, ML Ops |
> | Responsibility     | From business problem to data-driven solution | From algorithm design to system implementation |

#### Please name a few data science applications in areas such as healthcare, finance, e-commerce, and marketing.

> * **Healthcare**: 
	* **Disease diagnosis:** Analyze patients' medical data (such as medical history, imaging, genetic data) to diagnose disease in early stage. 
	* **AI-driven Drug Development:** Discover potential drug targets using AI to accelerate the drug development process and reduce development cost.
>* **Finance**: 
	* **Fraud Detection:** Identify fraudulent transactions and prevent financial fraud.
	* **Risk Assessment:** Evaluate credit risk using customer credit history, transaction records, and other data to provide decision support for loan approvals.
	* **Portfolio Optimization:** Construct optimal investment portfolio based on market data and client risk preferences. 
	* **Quantitative Trading:** Analyze market data to automatically carry out trading strategies.
>* **E-commerce**: 
	* **Recommendation System:** Recommend products of interest to users based on their historical purchase records, browsing behavior, and demographic data. 
	* **Demand Forecasting:** Predict future product demand to optimize inventory. 
	* **Customer Segmentation:** Segment customers based on different characteristics to achieve precise marketing.
	* **Price Optimization:** Dynamically adjust product prices based on market competition and product attributes.
> * **Marketing**: 
	* **Customer Relationship Management:** Understand customer needs, and improve customer satisfaction and loyalty. 
	* **Advertising:** Optimize advertising strategies to improve conversion or ROI.
	* **Measurement:** Measure the ROI of marketing activities to provide a basis for subsequent marketing decisions.

#### Provide 2-3 examples of outstanding data scientists around you, highlighting their exceptional qualities.

#### Considering your professional background and work experience, think about your career plan for the next 3-5 years. If you wish to transition to a data scientist or algorithm engineer, how should you identify and address gaps in your skills?

#### What technical skills and knowledge are essential for transitioning from small-scale data analysis to large-scale data science projects?

>* **Data Storage:** Distributed storage systems (e.g., Hadoop HDFS, AWS S3, Azure Blob Storage, NoSQL databases) and data lakes.
>* **Computation:** Big data processing frameworks (e.g., MapReduce, Spark) and parallel computing techniques.
>* **System Implementation:** Data pipelines, monitoring tools, and containerization technologies.

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 2 Statistical Foundations

#### Toss a fair coin until there is a head. Please calculate the distribution of the total number of tosses (including the last toss).

> The distribution of the total number of tosses follows a geometric distribution, with probability mass function $P(X=k) = (1/2)^k$.

#### Monty Hall problem: The game consists of three doors, behind one of which is a car, while the other two have goats. The contestant randomly selects one door. After the contestant selects a door, the host will open another door, revealing a goat inside. Should the contestant switch doors at this point? What are the probabilities of winning if they switch versus if they do not switch?

> In Monty Hall problem, the probability of winning by randomly selecting a door is $1/3$. Therefore, the probability of winning by not switching doors is also $1/3$. 而在初始选择不中的情况下，换门都都会中奖，故换门中奖的概率是$2/3$。

#### Given an unfair coin with a probability of landing heads up, $p$, how can we simulate a fair coin flip?？

> A common approach is to flip the coin twice. If the two flips result in the same outcome (both heads or both tails), discard the result and flip again. If the two flips result in different outcomes (one head and one tail), we can use the first flip as the outcome of the fair coin flip. The probability of getting a head or a tail in this scenario is 1/2.

#### Please use a real-life example to explain false positives and false negatives. 

> For instance, consider a new diagnostic test for a disease.
>* **False Positive**: A healthy individual tests positive for the disease, leading to unnecessary concern and potential further testing.
>* **False Negative**: A person with the disease tests negative, delaying diagnosis and potentially hindering timely treatment.

#### > What are some common sampling techniques used to select a subset from a finite population? Please provide 3-5 examples.

> Commonly used sampling methods include: 
>  - Sampling with replacement.
>  - Sampling without replacement.
>  - Stratified sampling.
>  - Multi-stage sampling. 
>  - Systematic sampling.

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 3 Machine Learning

#### Analyze the advantages and disadvantages of algorithms such as LR, RF, and GBDT, and consider how to implement LR, RF, and GBDT algorithms in a distributed environment.

For the advantages and disadvantages of algorithms, see Table 3-3 of this book. The decision trees in RF can be constructed in parallel, making parallelization easy to achieve. GBDT is a serial algorithm, but in a distributed environment, the feature computation and tree construction for each round can be parallelized. LR can achieve parallelization through parameter servers and asynchronous updates.

#### For a binary classification problem, randomly select one positive sample and one negative sample. AUC can be expressed as the probability that the predicted value of the positive sample is greater than that of the negative sample; please provide the derivation process.

Assume that the predicted scores of the positive samples follow the distribution $F_TS(s)$, and the predicted scores of the negative samples follow the distribution $F_T​(t)$. Random variables $S$ and $T$ follow distributions $FS(s)$ and $FT(t)$, respectively. The probability that the predicted value of the positive sample is greater than that of the negative sample is $P(S>T)$. When transforming the probability $P(S>T)$ into the area under the ROC curve, considering that the area under the ROC curve can be represented as the integral of a binary uniform distribution over $[0, 1]$, the proof can be completed using the probability integral transformation.

#### What is the difference between XGBoost and GBDT algorithms?

| Feature | GBDT | XGBoost |
|---|---|---| 
| Loss Function | Primarily uses square loss or exponential loss | Supports custom loss functions and incorporates second-order derivative information | 
| Regularization | No explicit regularization terms | Introduces L1 and L2 regularization terms to prevent overfitting | 
| Optimization Algorithm | Based on gradient descent | Based on second-order Taylor expansion for more accurate fitting of the objective function | 
| Handling Missing Values | No specific mechanism for handling missing values | Built-in strategies for handling missing values | 
| Parallel Computing | Supports parallel computing, but with relatively lower efficiency | Supports parallel computing and optimizes parallel computing efficiency | 
| System Optimization | Relatively fewer optimizations | Optimizations for cache, column sampling, etc. |

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXckKpd6QuemcLLCTmy_Uy5alF_wfUX1CHNuw2a-Sc8PDuzJyLOWfvTWsFdcGDFG79E3j37cOkZKweCNWZkXX7Mq29dk7fLoZKtpKDXFGHSVcWbELQaW1jtUN9lpflNjPpszrelV8FlhNU8GZxIUtGQZqzSs?key=LUNIL0RdK8QZOOvRcz6T7w)

#### What methods can be used to combat overfitting and underfitting issues?

Ensemble learning (Bagging, Boosting), model regularization, and regularization techniques in neural networks (Dropout and Early Stopping) are all methods to combat overfitting. When a model exhibits underfitting, feature crossing, more complex models, or searching for new features can be employed.

#### List common distance-based clustering and density-based clustering algorithm.  Please provide 3-5 examples.

Common distance-based clustering algorithms include K-Means and hierarchical clustering, while density-based clustering algorithms include DBSCAN, HDBSCAN, etc.

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 4 Neural Networks and Deep Learning

#### What are some commonly used activation functions? Please compare them in terms of computational complexity, sparsity, and gradient.

* **Sigmoid function**, which transforms input values into a range between 0 and 1, representing the probability of the positive class. The Sigmoid function is a smooth S-shaped curve suitable for binary classification, but the gradient may vanish when the input values are too large or too small.

* **tanh function**, which also transforms input values into a range between -1 and 1. It is similar to Sigmoid, but has a wider output range, which can sometimes better handle variations in data.

* **ReLU function**, which turns negative input values to 0 while keeping positive input values unchanged. The ReLU function is simple and effective, commonly used as the default activation function. It helps avoid the vanishing gradient problem and performs well in many cases.

* **Leaky ReLU function**, which is an improvement over ReLU. It adds a small slope to negative input values, avoiding the dying ReLU problem. Leaky ReLU allows some negative input values to pass through, maintaining a broader activation range, which contributes to the stability of the model.

#### What are the common regularization methods in deep learning? What are the differences between layer normalization and batch normalization?

Common deep learning regularization methods include Dropout and early stopping. Layer normalization normalizes across the feature dimensions of a single sample, independent of batch size. Batch normalization normalizes across the same feature dimensions of samples within a batch. If the input sequence length of the model is variable, or if normalization should not depend on batch size, then layer normalization is a better choice. If the model input size is fixed and normalization is desired to accelerate model training, then batch normalization is a good choice.

#### What are the application scenarios for one-to-one, one-to-many, and many-to-many in the input and output layers of a Recurrent Neural Network?

The one-to-one application scenario of recurrent neural networks is image classification, one-to-many scenarios such as image-to-text conversion, and many-to-many scenarios like text translation.

#### Why can transformers fit massive data better than recurrent neural networks and avoid gradient explosion?

Compared to RNNs, transformers have advantages such as strong parallel computing capability, strong long-distance dependency modeling ability, and the ability to avoid gradient vanishing/explosion when processing massive data. Transformers alleviate gradient explosion through layer normalization.

#### In which tasks have transformers achieved good results? What are the future development trends?

The Transformer model has powerful sequence modeling capabilities and has achieved significant results in various fields such as natural language processing, computer vision, and speech recognition. Future development trends include multimodal capabilities, larger-scale models, lower computational and inference costs, and integration with technologies such as reinforcement learning and knowledge graphs.

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 5 The data science workflow

#### How to choose between manual feature engineering and automated feature engineering, specifically in which situations to use manual feature engineering and in which situations to use automated feature engineering?

First, it depends on domain knowledge, whether one understands the physical meaning of the features; second, it depends on the number of features. Typically, when domain knowledge is rich, there is an in-depth understanding of the data, and the number of features is small, manual feature engineering is employed. In contrast, when domain knowledge is limited, or the data volume is large, the cost of manually designing features is too high, and there are low requirements for model interpretability, automated feature engineering is used.

#### How to bucket continuous features based on data distribution, and what are the advantages and disadvantages of distribution-based bucketing?

Usually, we bucket based on the quantiles of the data; the advantage is that the sample sizes in each bucket are uniform, while the disadvantage is that the bucket boundaries are not integer values and are affected by distribution drift.

#### Reflecting on past work, what data collection methods have you encountered or used?

Transaction data recorded by terminal devices, web data crawled from the internet, user behavior logs, data returned from car sensors, etc.

#### What are the common methods for handling missing values?

Common methods include (1) modeling the missing data mechanism, treating whether data is missing as a binary categorical variable for modeling; (2) using imputation algorithms such as mean imputation and regression imputation.

#### What are the common methods for detecting outliers? Common methods include statistical-based methods and density-based methods.

Statistical-based methods include:

1. **The 3$\sigma$ principle:** Assuming the data follows a normal distribution, data points that exceed the mean ± 3 standard deviations are considered outliers;
2. **Box plot method:** Determine the range of outliers through the upper and lower quartiles of the box plot and 1.5 times the interquartile range;
3. **Z-score method:** Standardize the data and calculate the Z-score for each data point, with points exceeding a certain threshold considered as outliers.

Density-based methods include:

1. **DBSCAN clustering:** Classifies data points into core points, border points, and noise points, where noise points can be considered as outliers;
2. **LOF (Local Outlier Factor):** Determines the degree of outlierness by calculating the ratio of the local density of each data point to the local density of its neighbors.

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 6 Data Storage and Processing

#### In the convenience store example, we need to store the transaction time, transaction user, quantity of goods, and price for each order, but allow users to return items they are not interested in. Please design a database system.

* Understanding Requirements: The order includes transaction time, transaction user, and multiple product items. Product items include product name, quantity, and unit price. Returns: Users can return some products in the order.

* Table Design:

1. **Order Table (orders)**: Includes order\_id (primary key): Order ID, customer\_id: Customer ID, order\_time: Order creation time, total\_amount: Total order amount, status: Order status (Pending Payment, Paid, Completed, Canceled);
2. **Order details table（order\_items）**: Includes item\_id (primary key): Order detail ID, order\_id (foreign key): Associated order table, product\_id: Product ID, quantity: Product quantity, unit\_price: Unit price;
3. **Product Table (products)**: Includes product\_id (primary key): Product ID, product\_name: Product name, category: Product category, price: Product price;
4. **Returns Record Table (returns)**: Includes return\_id (primary key): Return Record ID, order\_item\_id (foreign key): Associated Order Detail Table, return\_quantity: Return Quantity, return\_time: Return Time

#### Assuming we read the price of each transaction in streaming data, how can we calculate the median price?

The median price can be approximately calculated based on the frequency distribution by bucketing the prices and accumulating the frequency of each price range in the streaming data.

#### Some vector search packages can also be used for vector clustering and searching, so why do we still introduce a vector database?

Vector search packages are more suitable for small-scale, single-machine environments for vector data processing and do not provide vector persistence. Vector databases are more suitable for the storage and retrieval of vector data in large-scale, distributed environments, providing more comprehensive functionality and higher performance. Choosing the right tool depends on the specific application scenario: if the data volume is not large and performance requirements are not high, a vector search package may be sufficient. If the data volume is enormous and high performance and availability are required, a vector database is the better choice.

#### What is the differences between a data warehouse and a data lake?

See the comparison of data warehouses and data lakes in Table 6-7.

#### Given the attendance record of a class, along with the date, user ID, and whether they were present that day (1 for present, 0 for absent), please find all student IDs that have been absent for more than 3 consecutive days.

The core of the problem lies in how to identify continuous absence segments in attendance data. This is a classic 'Gaps and Islands' problem in SQL. The idea is to sort the absence dates for each student and then perform a difference operation; a difference result of 1 indicates that they belong to the same continuous absence segment.

[\[↑\] Back to top](#数据科学方法与实践思考题)

# Chapter 7 Data Science Technology Stack

#### Given an XGBoost model trained with Python, please describe how to deploy this model in a production environment. The main steps are as follows:

1. Use XGBoost's built-in methods, Pickle, or Joblib to save the trained model as a file.

2. In the production environment, use the corresponding library to load the saved model file.

3. Choose an appropriate deployment platform (local or cloud).

4. Service the model, such as building a RESTful API or gRPC service.

#### Please compare common methods for hyperparameter tuning of machine learning models (Greedy, GridSearchCV, and Bayesian).

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcth64AhIZS1bg2G_coE4gbOZmo1wlDPKiFxJEABGxvwDFOCZJwMV3MF66xalIIwe5DO0YztUvV0GFwTFfNvqVU8to45wgTDxEeL96gszCbB6AcCOaBhmX6DB78vPOAzAnVJ4zaSwUQXW3oRIqpgguPE7xw?key=LUNIL0RdK8QZOOvRcz6T7w)

#### Assuming we have a click-through rate prediction model that outputs predicted click-through rates between 0 and 1. Please briefly describe how to calculate the evaluation metrics for this model offline and in real-time. What issues arise in the calculation of evaluation metrics?

Cross-entropy can be chosen as the evaluation metric. Offline evaluation is conducted after the model training is completed, using an independent test set to assess model performance. Real-time evaluation is performed after the model is deployed in the online environment, assessing online traffic. Common computational issues include (1) selection bias: online traffic typically consists of users and ad combinations that are inclined to click, and it is possible to calculate separately on online traffic and random traffic; (2) Click delay: the delay between exposure and click affects the calculation of cross-entropy; (3) Sample leakage: ensuring that the test data is not used for model training.

#### Taking the comment quality scoring model as an example, briefly describe how to perform offline training and online deployment of the model, and list the possible technology stack used.

Offline Training Phase

* Data Preparation: Collect, clean, and label comment data, and divide the dataset.

* Model Construction: Choose an appropriate model architecture (such as TextCNN, LSTM, BERT), define the loss function and optimizer.

* Model Training: Using PyTorch for model training and evaluating model performance.

Online Deployment

* Model Saving: Save the trained model as a file.

* Deployment Platform Selection: Choose an appropriate deployment platform (such as TorchServe, TensorFlow Serving, cloud service platforms, containerization platforms).

* Model Service: Provide REST API, gRPC, and other interfaces for client calls.

Technology Stack

* PyTorch: Deep learning framework

* NLTK/spaCy：自然语言处理工具

* Flask/FastAPI：Web框架

* Docker: Containerization technology

* Kubernetes: Container orchestration platform

* Cloud Service Platforms: AWS SageMaker, Google Cloud AI Platform, etc.

[\[↑\] Back to top](#数据科学方法与实践思考题)

# Chapter 8 Product Analysis

#### Why has the overall average income level decreased, even though the average income level of every ethnic group in the United States has increased compared to ten years ago?

This is a typical example of Simpson's Paradox, considering the impact of immigration policies on the changing proportions of different ethnic groups in the United States.

#### What is selection bias? What are the common methods for eliminating selection bias to obtain scientific conclusions?

Selection bias refers to the presence of bias in the selection process of experimental samples, leading to a lack of representativeness in the samples. Common types of selection bias include survivor bias, response bias, truncation bias, and confirmation bias. During the research design phase, random sampling and random assignment of experimental conditions can ensure that the samples are representative, and selecting appropriate control groups can enhance the credibility of the research results. During the analysis phase, methods such as post-stratification, regression analysis, or propensity score matching (PSM) can be used to reduce the impact of selection bias.

#### Please describe the registration funnel of a product you frequently use, propose ideas for optimizing the new user registration process, and validate them with data.

Refer to the registration funnel of ride-hailing drivers in Figure 8-13.

#### Please provide an analytical approach to the supply and demand relationship of content (such as short videos or graphic content) on social platforms.

* Taking short videos as an example

* Supply-side analysis:

1. Daily upload quantity: Count the number of short videos uploaded each day and analyze the trend of upload volume changes.

2. Vertical field analysis: Classify short video content according to different vertical fields (such as food, travel, education, etc.), count the upload quantity in each field, and identify popular and niche areas.

3. Video Quality Analysis: Classify videos into different quality levels based on metrics such as clarity, editing quality, and originality of content, and count the number of uploads for each quality level.

* Demand Side Analysis:

1. View Count Statistics: Count the daily view count of short videos and analyze the trend of view count changes.

2. Interaction Metrics: Analyze user demand for different videos based on user interaction data such as likes, comments, and shares.

3. User Profile: Analyze which types of users prefer certain categories of short video content based on information such as gender, age, and region.

The above is the basic idea for analyzing the supply and demand relationship of short video content on social platforms.

#### The GMV (Gross Merchandise Value) of a certain e-commerce platform has increased by 8% year-on-year. Please analyze the possible reasons behind this based on data.

Multiple group analyses can assist in identifying the underlying reasons:

* Based on user segmentation: conduct segmentation analysis by user gender, geographical location, new vs. old users, etc.;

* Based on product segmentation: analyze by product category and check whether popular products have driven the overall GMV;

* Time series analysis: consider whether the impact is due to monthly or weekly trends, or specific holiday effects.

* Transaction page source analysis: break down by page (homepage, category page, search results page) to see if there are changes in transactions brought by different pages;

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 9 Metrics

#### Assuming we consider that the subscription-based Netflix streaming platform increased its subscriber count by 6% by lowering the monthly fee from $11.99 to $9.99. How should we evaluate whether the monthly fee should be reduced?

The main focus is on metric formulation and selection. If revenue from subscription users is used as a measure, the growth of subscription users is insufficient to offset the loss from declining monthly fees. In actual decision-making, cost factors, other businesses such as advertising revenue, and the impact on long-term income will also be considered.

#### How can we use a metric system to measure the quality of content viewed by users on platforms like Toutiao?

1. User interaction metrics: Click-Through Rate (CTR), reading duration, likes, comments, shares, etc.
2. Content quality metrics: Content relevance, freshness, richness.
3. User feedback metrics: User satisfaction surveys, negative feedback.
4. Recommendation Metric: Content Distribution Efficiency.

#### How to set target metrics for short video platforms such as Douyin?

From a macro perspective, consider user growth metrics (DAU, MAU, user retention rate) and revenue metrics (advertising revenue, e-commerce revenue, proportion of paying users). For content communities, attention will also be paid to content creation metrics (number of content creations, number of creators, number of creators earning revenue) and user engagement metrics (playback, likes, comments, etc.).

#### What metrics are used to measure the relevance of search results?

The relevance of search results can be measured through subjective evaluation and objective evaluation. Subjective evaluation requires the establishment of evaluation standards, labeling results through manual evaluation or large language models, and calculating metrics such as NDCG. Objective evaluation can be conducted through user behavior metrics such as click-through rate, long play rate, and bounce rate.

#### If we suppress low-quality content on the platform, it may lead to a decline in user activity levels and even some user attrition. In this case, how should product decisions be made?

* When the platform suppresses low-quality content, it faces the risk of declining user activity levels and creator attrition, which requires a comprehensive consideration of multiple dimensions such as platform tone, revenue metrics, and creator impact.

* Clarify the platform's long-term goals: Should it emphasize high-quality content to enhance user satisfaction and retention, or prioritize quantity for rapid growth? Is the aim to create a high-quality, valuable platform, or to focus more on entertainment and diversity?

* Assessing the impact of low-quality content on the platform: Quantifying the effects of low-quality content on user duration and interaction metrics through data analysis and low-quality content filtering experiments;

* Considering various factors: Taking into account the platform's tone, the impact on revenue metrics, and the effect on creators (whether creators of low-quality content will churn or shift towards content encouraged by the platform) to formulate response strategies.

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 10 A/B Testing

#### What could cause sample ratio mismatch (SRM) in A/B testing, and what governance measures are available?

SRM may arise from issues with the random generator (users, sessions, page views, clicks, etc. not being allocated in a truly random manner from the outset), technical problems in experiment delivery (different probabilities of seeing the test variant), issues with log collection (different probabilities of data collection), or reporting errors. In addition, measures taken to ensure data quality, such as bot filtering, can also impact user allocation between experimental groups. The SRM problem can be addressed through SRM ratio monitoring and bucket uniformity verification to ensure the reliability of experimental data, and more scientifically valid experimental conclusions can be obtained through methods such as controlling variables and PSM (Propensity Score Matching).

#### Let us consider the scenario of new user registration and focus on the number of user behaviors on the platform after registration. In this scenario, should A/B testing use user segmentation or device segmentation?

In this scenario, it is recommended to use device segmentation. Device IDs already exist before user registration and can serve as unique identifiers for assigning experimental groups. However, the limitations of a single user using multiple devices must also be considered.

#### Please introduce Poorman’s bootstrap and write the pseudocode for executing Poorman’s bootstrap.

There is no unified definition of Poorman’s bootstrap in the literature; in the context of A/B testing, Poorman’s bootstrap is a method to address the efficiency loss caused by sampling with replacement. This method buckets the data, calculates the metrics of interest for each bucket, and computes the variance by examining the variance between the metrics of the buckets.

#### What are common methods to reduce the fluctuation of experimental metrics?

Common methods to reduce the fluctuation of experimental metrics include metric transformation, truncation, and outlier removal, which are relatively lightweight yet effective methods. Additionally, controlling variables can reduce metric fluctuations, such as CUPED or stratified sampling.

#### Please recall the feedback on the A/B testing platforms used in past work. What level is the company's experimental culture at (crawling, walking, running, or flying)?

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 11 Search Recommendation Advertising Models

#### The two main ways users acquire information are through search and recommendation. Please explain the differences between search and recommendation from the perspectives of user intent, business objectives, user profiles, and personalization.

See Table 11-1 for details.

#### Please write out the formula for TF-IDF and explain its meaning.

The formula for TF-IDF is as follows:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeTQYLgfJBbV8D4nCyZXM0VI0r2PJuhGdMcpb8HYcLI-LDhx5lnotCHD7t1i_f91tJ-_C25et3y_-cUOn0nKktylNUZR7ISb9iSUUjg5enFjXkr3g4YnUvfytNS2fWfdHdJIwZsW8xlpDe645IIig-mW-0L?key=LUNIL0RdK8QZOOvRcz6T7w)

Where TF has multiple variants, which can use raw frequency, relative frequency, or logarithmic frequency. IDF (Inverse Document Frequency) is typically calculated using the following formula:

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfuEMJWxdyVZmPC0kgwIFH0oR-rvEMNWOdhQTpb3WA_jao2ITFnq6bvobb73btpUofWzCUrU5BN4hclH8yWSKAp9auAxYz9UcVdlP4Ojdziih82YK_U-O-d20Trc_tyPhm_bOgGHyWQEgHcl89Y53EbqKqX?key=LUNIL0RdK8QZOOvRcz6T7w)

#### Please describe the steps of the PageRank algorithm.

The implementation steps of the PageRank algorithm are roughly as follows:

1. 假设有n个网页，初始化每个网页的PageRank值为1/n。

2. Ignore self-links or multiple links between the same webpage. Calculate the out-degree of each webpage (i.e., the number of links to other webpages).

3. Choose a damping factor d (usually 0.85).

4. Iteratively update the PageRank value of each webpage until convergence or the maximum number of iterations is reached. The update formula is shown below, where R is the n-dimensional PageRank value vector, M is the state transition matrix, and 1 represents an n-dimensional vector with all elements equal to 1.

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdklc9Ba2m-QZ3njNy06pBzydA_wklqINGmV3lx-qOizh5m4h3DjgaR94QfVIV_x7nXhAvqsXjDq2bqzlcMgWiZYEBgdFBd6_szQ5wXBdCCUWDY3s2ggjDgQU5zPA1gdVJQpnEJ0XNUbGRlmNiCWDdOXGpz?key=LUNIL0RdK8QZOOvRcz6T7w)

#### Please describe the offline training and online service process of the dual-tower neural network model.

* The dual-tower neural network model is widely used in recommendation systems. The dual-tower neural network model constructs a user tower and an item tower during model training, transforming raw features into dense vector representations, and uses negative sampling to construct negative samples. Online services focus on model deployment and real-time prediction. In response to user requests, extract the user feature vector, calculate the similarity between the user vector and all item vectors, and return the TopN recommendation results.

#### What are the differences between the FTRL algorithm and other online learning algorithms such as SGD in terms of regularization methods, learning rate adjustment, historical information usage, and application scenarios?

| Feature | SGD | FTRL |
|---|---|---| 
| Regularization | L1/L2 | Adaptive L1+L2 | 
| Learning Rate | Fixed or Decaying | Adaptive | 
| Historical Information | Uses less | Uses fully | 
| Use Cases | General | Sparse Data |

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 12: Recommender Systems

#### What metrics does the recommendation recall model use for evaluation?

When evaluating, it is necessary to consider the individual effect of each recall source and the net benefit compared to other recall sources. Individual evaluation: Typically conducted offline, the metrics used include top K accuracy, recall rate, F1 score, and hit rate, among others. Comparative evaluation is based on the premise of having other recall sources, assessing the additional gain from this recall source. Offline evaluation can be conducted through metrics such as recall overlap and consistency with the results of the precision ranking model.

#### Please describe the recall, ranking, and re-ranking modules of the recommendation system, including the function of each module and the models used.

See section 12.2.

#### In the context of information flow recommendation, assuming the business goal is to improve user retention, how should one choose appropriate recommendation ranking metrics?

A common practice is to use user retention as the dependent variable and user interaction metrics such as playback, likes, and comments as independent variables for regression, referencing the weights of the regression model to formulate appropriate recommendation ranking metrics.

#### How can we enhance the diversity of recommendation results?

Firstly, through post-processing, and secondly, through list-level reordering. Post-processing involves adjusting the ranking results during the fine-tuning stage, such as dispersing duplicate content within adjacent or sliding windows to enhance diversity. List-level reordering can utilize methods based on Determinantal Point Processes (DPP) to optimize list-level diversity.

#### What dimensions are typically included in the user profile of a recommendation system, and how can a user profile platform be constructed?

Common dimensions of profile features: demographic characteristics, interest tags, user behavior history.
The construction of the profiling platform includes (1) building a tagging system and using machine learning algorithms to train the interest tag model; (2) Visualization: providing a user-friendly interface to visually display user profile information; (3) API Interface: providing an API interface for other systems to conveniently access user profile data. (4) Application Scenarios: identifying application scenarios within enterprises for personalized recommendations, precision marketing, user segmentation, etc.

[\[↑\] Back to top](#数据科学方法与实践思考题)

# Chapter 13 Application Scenarios in the Advertising Field

#### What are the differences between brand advertising and performance advertising in terms of purpose, duration, optimization goals, and revenue evaluation?

See Table 13-1.

#### Assuming that the advertiser has a budget that cannot be spent and cannot obtain exposure traffic, how should we diagnose this situation?

Diagnosis includes the following: 

 - Targeting issues: Check if the targeting is too narrow or affected by
   filtering rules;         
 - Bid competitiveness issues: Is the bid too
   low and    lacking competitiveness; 
 - Ad creative appealingness:
   Is the ad    creative lacking appeal; 
 - Ad account issues: Is the
   account suspended    or restricted? Has the account set a daily
   budget limit that has been    reached?

#### Please describe the business process of advertising on social media platforms such as Facebook, including the entire process of ad creation, budget and bid setting, ad targeting, ad display, and click conversion.

The following is the business process of advertising on social media platforms (taking Facebook as an example):
1. Ad Creation: This includes selecting advertising objectives (such as increasing brand awareness, driving website traffic, generating leads, etc.), creating ad groups, designing ad materials, etc.;
2. Budget and Bid Setting: Set a daily budget, choose a bidding method (cost-per-click, cost-per-impression, automated bidding, etc.), and bidding strategy;
3. Ad Targeting;
4. Ad Display;
5. Ad Conversion: tracking user behavior after clicking on the ad;

#### What are the common budget pacing methods?

Common methods include
 - adjusting bids: enhancing ad spending by modifying bids;
 - Probabilistic throttling: control budget consumption through the probability of
   participation.

#### What problem does the budget isolation experimental mechanism aim to solve? How is a budget isolation A/B test typically created?

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 14 Application Scenarios in Search

#### Please describe the process of constructing an inverted index.

See Figure 14-3 for the process of generating an inverted index: tokenization, text processing, and indexing.

#### Please write out the formula for the NDCG metric and provide an example of how NDCG is calculated.

See section 14.3.2 of this book.

#### How do you evaluate the relevance and authority of search results?

The relevance of search results can be measured through subjective evaluation and objective evaluation. Subjective evaluation requires the establishment of evaluation standards, labeling results through manual evaluation or large language models, and calculating metrics such as NDCG. Objective evaluation can be conducted through user behavior metrics such as click-through rate, long play rate, and bounce rate. Evaluating the authority of search results generally focuses on website authority, author authority, and author-category authority, where author-category authority indicates the author's authority within the current category.

#### What are the common classification methods for search terms?

Classified as general types into broad intent and precise search, by timeliness into timely and non-timely, and by intent into navigational intent, information retrieval, and transactional intent.

#### What are the advantages and disadvantages of subjective and objective evaluations of search? Compare subjective and objective evaluations of search in terms of distinguishability, evaluation accuracy, applicability, and evaluation cost.

Subjective evaluation usually can only categorize whether or not or into a few simple levels, with low distinguishability. The accuracy of subjective evaluation is affected by the evaluator's grasp of user intent, while the accuracy of objective evaluation is influenced by noise in user behavior data. Subjective evaluation is applicable to end-to-end evaluation of search engines, as well as to the evaluation of individual modules such as the intent understanding module. In terms of evaluation cost, subjective evaluation usually requires hiring people for assessment, while objective evaluation requires development and deployment, as well as a large number of A/B testing iterations.

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 15 Natural Language Models and Application Scenarios

#### Please explain the negative sampling and hierarchical SoftMax methods of the Word2Vec model.

See Section 15.3.2 of this book.

#### How can sentence embeddings be obtained from word embeddings?

A simple method is to take the average of the word embeddings in the sentence, or to weight the word vectors; a simple and easy-to-use modification method is the SIF embedding proposed by Arora et al. in 2017.

#### What does token mean in Natural Language Processing, and what are some common tokenizers?

In Natural Language Processing, a token can be simply interpreted as the smallest semantic unit in textual data. It can be a word, a subword, or even a punctuation mark. Table 15-3 of the book introduces common tokenization methods.

#### What are the similarities and differences between LSTM and transformer models in Natural Language Processing?

Both LSTM and transformer models can be used to process sequential data; compared to LSTM, transformers can better parallelize and are more adept at handling long-distance dependencies.

#### Please describe the development context of Natural Language Processing and typical scenarios of Natural Language Processing.

See Chapters 15.1 and 15.2 of this book for details.

[\[↑\] Back to top](#数据科学方法与实践思考题)

## Chapter 16 Large Language Models

#### What are the three dimensions that characterize the scale of LLMs?
 - Number of model parameters.
 - Number of training tokens
 - Computational resources required for training.

#### How are intrinsic hallucination and extrinsic hallucination defined, and what are some ways to eliminate hallucinations?

 - **Intrinsic hallucination:** The generated text is logically contradictory to the source content. 
 - **Extrinsic hallucination:** We cannot verify the correctness of the output from the provided source; The source content does not provide enough information to assess the output, making it uncertain.

Methods to reduce hallucinations in large model outputs include decoding strategy optimization and retrieval-augmented generation.

#### Please describe the workflow of Retrieval-Augmented Generation (RAG).

Retrieval-Augmented Generation mainly consists of the following steps:

1. **Index Construction:** The process of index construction includes data preprocessing, text segmentation, and vectorization.
2. **Retrieval:** Common retrieval methods include keyword retrieval, vector retrieval, knowledge graph-enhanced retrieval, and advanced retrieval methods.
3. **Generation:** During the generation process, we focus on the base model used for generation and controlling the quality of the generation.

#### What are the data challenges of LLMs?

The data challenges of large models include the challenges of storage, cleaning, and training brought by massive data, with data cleaning challenges manifested in data redundancy, traversal, and PII leakage, among others.

#### Please describe the process of fine-tuning a LLM.

The fine-tuning of a large model mainly involves the following steps:
1. **Data Preparation:** Collect and organize the data we want the model to learn from, and label this data correctly.
2. **Model Selection:** Choose a suitable large model for the task, such as BERT, GPT, etc.
3. **Model Training:** Adjust the model's learning rate, batch size, and other parameters to train the model with the prepared data.
4. **Model Evaluation:** Evaluate the performance of the fine-tuned model.
5. **Model Saving:** Save the trained model for easy access in subsequent tasks.