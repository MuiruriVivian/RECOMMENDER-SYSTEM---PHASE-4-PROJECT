# BUILDING A PERSONALIZED MOVIE RECOMMENDATION SYSTEM
---
COLLABORATIVE AND CONTENT-BASED FILTERING 

![Sample Image](AAAAQRC29H19twWKcTZ9Zpg4biJbGNaHF2GGIYNcLt4eZ6fvwugUJbuKxTjjMFPCS-y5P3ZePL57rupDtSkyUIJhv3P8leMJGMzszuG2CHNd65NwWPu5LeKxQkRNfNMHmxAwt7tmQZFk1VIrBd1aXr2AR5DM.jpg)

## BUSINESS PROBLEM:
---


In today's digital world, users are overwhelmed with vast amounts of content, whether it's movies, products, music, or news articles. Businesses struggle to keep users engaged by providing personalized recommendations. Without an effective recommender system, customers may churn, engagement may decline, and businesses may lose revenue opportunities.


For example, in an online movie streaming platform, users need relevant and personalized movie recommendations based on their viewing history and preferences. A poor recommendation system may result in users struggling to find interesting content, leading to lower customer satisfaction and reduced subscription retention.

## OBJECTIVES 
---


By implementing an effective recommender system, businesses can:

1. Increase user engagement and retention. 

2. Improve customer satisfaction by offering relevant recommendations.

3. Enhance revenue opportunities through personalized marketing.



## GOALS
---

The goal of this study is to develop a personalized recommendation system that improves user experience and engagement by suggesting relevant content based on past interactions. This will be achieved using:

1. Collaborative Filtering: Predict user preferences based on similar users.

2. Content-Based Filtering: Recommend items similar to what a user has liked before.

3. Hybrid Approach: Combine multiple recommendation techniques for better accuracy.

4. Performance Evaluation: Assess the effectiveness of different models using evaluation metrics such as RMSE (Root Mean Squared Error) and Cosine Similarities 

## IMPORTING THE NECESSARY LIBRARIES

Import necessary libraries for data handling, visualization, and modeling. The imported packages are:
1. pandas
2. matplotlib
3. seaborn
4. numpy
5. sklearn
6. warnings
7. suprise

## LOADING THE DATASET

* This is Loading the necessary datasets (e.g., user-item interactions, item metadata).

* The Data used was **MovieLens Latest Datasets** which was extracted from https://grouplens.org/datasets/movielens/latest/. 

* The data consist of 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.

* The Dataset has links, movies and rating data that was merged as one dataset. 

## DATA CLEANING

* Inspect and clean the data which will be done as follows:

    1. Merging the Datasets

    2. Drop the irrelevant columns

    3. Handle missing data.

    4. Remove duplicate records

    5. Clean or transform data types as necessary.

## PRE-PROCESSING AND FEATURE ENGINEERING

* This was conducted in order to transform raw data into a structured format suitable for machine learning models.

    1. Extracting year from movie titles.

    2. Converting Data into Model-Specific Format

    3. Splitting Data for Training & Testing

## EXPLORATORY DATA ANALYSIS (EDA)

* Exploratory data analysis (EDA) was performed to understand the distributions, correlations, and patterns.


**EXPLAINATION**
According to the EDA:

1. Rating
    The histogram shows that the majority of the movies are rated 4 while very few had a rating of 1.

    The distribution is skewed towards higher ratings, with the highest frequency at Rating 4.

    As the ratings increase from 1 to 5, the number of counts increases, showing that more movies received higher ratings.

    This pattern suggests that, in this dataset, movies are more likely to receive higher ratings than lower ones.

2. Genre:
    Drama is the most frequent genre, with over 40,000 movies. It is followed closely by Comedy and Action, which also have large counts.

    Genres such as Thriller, Adventure, and Romance appear in the middle range, with counts significantly lower than Drama but still in the high teens to low 20,000s.

    On the other end of the spectrum, genres like Western, Documentary, and Film-Noir are much less frequent, with counts well under 5000.

3. Rating by Title
    
    The top 5 Most rated movies are Hollywood Chainsaw Hookers, Calcium Kid,Chinese Puzzle (Casse-tête chinois), Raise Your Voice and Rain. 

    However, the basic statistics shows that the poorly rated movies are Indestructible Man,Yongary: Monster from the Deep,  Don't Look Now, Superfast! and Anaconda: The Offspring

4. Rating per Genre
    The genres that are highly rated are Film-Noir, War, Documentary, Crime and Drama. 

    On the other hand, The low rated genre were Horror,Comedy, Children,Actions and Sci-Fi


## BUILD THE RECOMMENDER SYSTEM

### A. Collaborative Filtering Using Surprise

### B. Content-Based Filtering (Using Cosine Similarity)

### C. Hybrid Based Flitering


MODEL EVALUATION

MAKING A PREDICTION

Predict ratings for a user-movie pair.

MAKE RECOMMENDATION BASED ON A MOVIE TITLE

MAKE RECOMMENDATION BASED ON GENRE

Explode Genres and Count Views per Genre
