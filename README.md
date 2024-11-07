# Supplementary Materials for ***Data Science Methods and Practice***

This repo contains supplementary materials for the book *Data Science Methods and Practice*, published by China Machine Press in 2024/2025. 

<img src="https://github.com/qqwjq1981/data_science_practice/blob/main/dataScience.jpg" alt="Alt text" width="500" height="750">


This repo is organized as follows:
- Data Science Interview Questions & Practical Exercises.
- Case study A: Personalized recommendation based on Linkedin profile.
- Case study B: Data pipeline for a conceptual convenience store example.

### Data Science Interview Questions & Practical Exercises


### Personalized recommendation based on Linkedin Profile
The overall project aims to create a system that provides users with personalized content recommendations. It achieves this by collecting relevant articles from various websites based on predefined topics (**RSS Crawling**), using advanced techniques like embeddings and clustering to group similar articles together, (**Embedding and Clustering**) and analyzing a user's LinkedIn profile to understand their interests (**Profile Understanding and Recommendation**).

#### RSS Crawling
The crawling module is responsible for crawling from RSS-enabled websites and saving the crawled content, with the following steps:

1. Determine mapping between predefined interests and RSS-enabled websites.
2. Crawl the RSS feeds of the identified websites on a daily or weekly basis and save the crawled content.

#### Embedding and Clustering
This module processes the content gathered by the crawling module and organizes it using embeddings and clustering techniques, with the following steps:

1. Generate an embedding representation by calling OpenAI API.
2. Apply clustering algorithm to group similar articles together based on their embedding representations.
3. Load the clustering and embedding representations into Weaviate, a vector database.
#### Profile Understanding and Recommendation
This module focuses on understanding a user's interests based on their LinkedIn profile and provides personalized recommendations:
 1. Use a large language model like GPT to analyze users’ LinkedIn profiles and identify their professional interests.
2. Use Weaviate’s hybrid search approach to provide personalized offline recommendations.

### Data pipeline for conceptual convenience store example

(used to introduce data schema, real-time and offline data flow, and transaction analysis)

# 数据科学方法与实践配套资源

本 repo 包含由机械工业出版社于 2024/2025 年出版的书籍 [数据科学方法与实践](https://) 的配套资源。本 repo 组织如下：
- 书中思考题的详细答案或提示。
- 案例研究 A：基于 Linkedin 个人资料的个性化内容推荐。
- 案例研究 B：便利店示例的数据管道。

### 基于 Linkedin 个人资料的个性化内容推荐
整个项目旨在创建一个为用户提供个性化内容推荐的系统。它根据预定义主题从各个网站抓取相关文章（**RSS 抓取**）、使用嵌入和聚类等技术将相似的文章聚合在一起（**内容嵌入和聚类**），分析用户的 LinkedIn 个人资料了解用户画像并进行推荐（**用户理解和推荐**）。
#### RSS 抓取
RSS抓取模块负责从支持 RSS 的网站抓取内容，步骤如下：
1. 确定兴趣主题与支持 RSS 的热门网站之间的映射。
2. 抓取已识别网站的 RSS 源，并保存抓取的文章。

#### 内容嵌入和聚类

本模块处理抓取的内容并使用嵌入和聚类技术对其进行组织，步骤如下：
1. 调用 OpenAI API 生成嵌入表示。嵌入是文本的数字表示，可捕获内容的语义含义。
2. 应用聚类算法，根据相似文章的嵌入表示将它们分组在一起。
3. 将嵌入和聚类结果加载到向量数据库 Weaviate 中。

#### 用户理解和推荐
本模块根据用户的 LinkedIn 个人资料理解用户兴趣并进行推荐，步骤如下：
1. 使用大型语言模型 GPT 分析用户的 LinkedIn 个人资料并确定他们的专业兴趣。
2. 使用 Wea​​viate 的混合搜索方法提供个性化的离线推荐。

### 便利店示例的数据管道

（用于介绍数据模式、实时和离线数据流以及交易分析）