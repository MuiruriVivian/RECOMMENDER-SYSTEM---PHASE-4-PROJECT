# BUILDING A PERSONALIZED MOVIE RECOMMENDATION SYSTEM
---
COLLABORATIVE AND CONTENT-BASED FILTERING 
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

IMPORTING THE NECESSARY LIBRARIES

Import necessary libraries for data handling, visualization, and modeling.

LOADING THE DATASET

Read and explore the dataset.

DATA CLEANING

Merging the Datasets

Drop the irrelevant columns

Handle missing data.

Remove duplicate records, if any.

Clean or transform data types as necessary.

DATA PREPROCESSING

This is conducted in order to transform raw data into a structured format suitable for machine learning models.

Extracting year from movie titles.

Converting Data into Model-Specific Format

Splitting Data for Training & Testing

EXPLORATORY DATA ANALYSIS (EDA)

Basic statistics.

Data visualization.

1. Rating

EXPLANATION

The histogram shows that the majority of the movies are rated 4 while very few had a rating of 1.

The distribution is skewed towards higher ratings, with the highest frequency at Rating 4.

As the ratings increase from 1 to 5, the number of counts increases, showing that more movies received higher ratings.

This pattern suggests that, in this dataset, movies are more likely to receive higher ratings than lower ones.

2. Genre:

EXPLANATION

Drama is the most frequent genre, with over 40,000 movies. It is followed closely by Comedy and Action, which also have large counts.

Genres such as Thriller, Adventure, and Romance appear in the middle range, with counts significantly lower than Drama but still in the high teens to low 20,000s.

On the other end of the spectrum, genres like Western, Documentary, and Film-Noir are much less frequent, with counts well under 5,000.

3. Rating by Title

4. Rating per Genre

BUILD THE RECOMMENDER SYSTEM

A. Collaborative Filtering Using Surprise

B. Content-Based Filtering (Using Cosine Similarity)

MODEL EVALUATION

MAKING A PREDICTION

Predict ratings for a user-movie pair.

MAKE RECOMMENDATION BASED ON A MOVIE TITLE

MAKE RECOMMENDATION BASED ON GENRE

Explode Genres and Count Views per Genre