# BUILDING A PERSONALIZED MOVIE RECOMMENDATION SYSTEM
---
COLLABORATIVE AND CONTENT-BASED FILTERING 

![Movies](https://github.com/MuiruriVivian/RECOMMENDER-SYSTEM---PHASE-4-PROJECT/blob/main/image/movies.jpg) 



## 📌BUSINESS PROBLEM:
---


In today's digital world, users are overwhelmed with vast amounts of content, whether it's movies, products, music, or news articles. Businesses struggle to keep users engaged by providing personalized recommendations. Without an effective recommender system, customers may churn, engagement may decline, and businesses may lose revenue opportunities.


For example, in an online movie streaming platform, users need relevant and personalized movie recommendations based on their viewing history and preferences. A poor recommendation system may result in users struggling to find interesting content, leading to lower customer satisfaction and reduced subscription retention.

## 📌 OBJECTIVES 
---


By implementing an effective recommender system, businesses can:

1. Increase user engagement and retention. 

2. Improve customer satisfaction by offering relevant recommendations.

3. Enhance revenue opportunities through personalized marketing.



## 📌 GOALS
---

The goal of this study is to develop a personalized recommendation system that improves user experience and engagement by suggesting relevant content based on past interactions. This will be achieved using:

1. Collaborative Filtering: Predict user preferences based on similar users.

2. Content-Based Filtering: Recommend items similar to what a user has liked before.

3. Create a model and carry out a model evaluation

4. Performance Evaluation: Assess the effectiveness of different models using evaluation metrics such as RMSE (Root Mean Squared Error) and Cosine Similarities 

## 📌IMPORTING THE NECESSARY LIBRARIES
---

Import necessary libraries for data handling, visualization, and modeling. The imported packages are:
1. pandas
2. matplotlib
3. seaborn
4. numpy
5. sklearn
6. warnings
7. suprise

## 📌 LOADING THE DATASET
---

* This is Loading the necessary datasets (e.g., user-item interactions, item metadata).

* The Data used was **MovieLens Latest Datasets** which was extracted from https://grouplens.org/datasets/movielens/latest/. 

* The data consist of 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.

* The Dataset has links, movies and rating data that was merged as one dataset. 

## 📌 DATA CLEANING
---

* Inspect and clean the data which will be done as follows:

    1. Merging the Datasets

    2. Drop the irrelevant columns

    3. Handle missing data.

    4. Remove duplicate records

    5. Clean or transform data types as necessary.

* After cleaning the dataset has 100836 rows and 6 Columns where the relevant columns are **movieId**, **title**, **genre**, **userId** and **Rating**. 

## 📌PRE-PROCESSING AND FEATURE ENGINEERING
---

* This was conducted in order to transform raw data into a structured format suitable for machine learning models.

    1. Extracting year from movie titles.

    2. Converting Data into Model-Specific Format

    3. Splitting Data for Training & Testing

## 📌EXPLORATORY DATA ANALYSIS (EDA)
---

* Exploratory data analysis (EDA) was performed to understand the distributions, correlations, and patterns.

---

**Visualization**

Genre Distribution

![Genre_Distribution](https://github.com/MuiruriVivian/RECOMMENDER-SYSTEM---PHASE-4-PROJECT/blob/main/image/Genre%20Distribution.jpg)

Rating Distribution

![Rating_Distribution](https://github.com/MuiruriVivian/RECOMMENDER-SYSTEM---PHASE-4-PROJECT/blob/main/image/Rating%20Distribution.jpg)

---

**EXPLAINATION**
According to the EDA:

1. Rating:

    The histogram shows that the majority of the movies are rated 4 while very few had a rating of 1. The distribution is skewed towards higher ratings, with the highest frequency at Rating 4. 

    As the ratings increase from 1 to 5, the number of counts increases, showing that more movies received higher ratings. This pattern suggests that, in this dataset, movies are more likely to receive higher ratings than lower ones.

2. Genre:

    Drama is the most frequent watched genre, with over 40,000 movies. It is followed closely by Comedy and Action, which also have large counts.

    Genres such as Thriller, Adventure, and Romance appear in the middle range, with counts significantly lower than Drama but still in the high teens to low 20,000s.

3. Rating by Title
    
    The top 5 Most rated movies are Hollywood Chainsaw Hookers, Calcium Kid,Chinese Puzzle (Casse-tête chinois), Raise Your Voice and Rain.  In the recommender system, these movies likely had a significant influence on recommendations, especially in collaborative filtering models.

    However, if the system relies on content-based filtering, recommendations will be influenced by the features (such as genres or plot similarities) rather than just the number of ratings.

4. Rating per Genre

    The genres that are highly rated are Film-Noir, War, Documentary, Crime and Drama. 
    
    If many users have highly rated movies from these genres, the system will likely recommend similar genres to users with matching preferences. 

    Users who have rated movies in these genres highly will receive recommendations for other films within the same genres, based on similarities in user rating patterns.

    On the other hand, The low rated genre were Horror,Comedy, Children,Actions and Sci-Fi


## 📌 BUILD THE RECOMMENDER SYSTEM
---

### Collaborative Filtering
---

Recommendation model is Build using **item-based collaborative filtering** and **user-based collaborative filtering**. This approach recommends items (movies) and users based on the similarities between them.

The Features used were:

- **Collaborative Filtering** using SVD  
- **TF-IDF Vectorization** for genre-based recommendations  
- **RMSE Evaluation** for performance measurement  
- **Predicts User Ratings** for unseen movies  
- **Top-N Movie Recommendations** for users  

item-based collaborative filtering:

- In item-based collaborative filtering, the system recommends items that are similar to the ones the user has already liked.

- For example, the highly rated movie is "Hollywood Chainsaw Hookers" thus the system will recommend movies that other users who liked "Hollywood Chainsaw Hookers" also enjoyed.

user-based collaborative filtering

- In user-based collaborative filtering, the system recommends movies based on the preferences of users who have similar tastes. It identifies users with similar rating patterns and suggests movies that those users have liked but the target user hasn't seen yet.

- For example, if a user highly rated "Hollywood Chainsaw Hookers", the system will look for other users who also liked this movie. If those users also rated "Calcium Kid" and "Chinese Puzzle" highly, then these movies will be recommended to the target user.
  
---
### Content Based Filtering 
---

 Content-Based Movie Recommender System suggests movies based on their features such as genres, descriptions, and other metadata. The system utilizes TF-IDF vectorization and cosine similarity to find movies similar to a given input movie.

The Features used were:

- Content-Based Filtering: Recommends movies based on their similarity in genres and descriptions.

- TF-IDF Vectorization: Converts textual movie data into numerical form.

- Cosine Similarity: Measures the similarity between movies based on their features.

- Customizable Recommendations: Users can input a movie name to get personalized recommendations.

## 📌 Model Evaluation
---

- The model achieved an RMSE of 0.8748, meaning the predicted ratings deviate from the actual ratings by approximately 0.87 on average. 

- Since RMSE is a measure of error, a lower value indicates better prediction accuracy. 

- However, an RMSE close to 1 suggests that while the model performs reasonably well, there is still room for improvement through hyperparameter tuning, incorporating additional features, or using a more advanced recommendation techniqu

## 📌 Making a Prediction 
---

- The performance is evaluated using Root Mean Squared Error (RMSE), which measures the difference between predicted and actual ratings. 

- The resulting RMSE is 0.9734, indicating that, on average, the model's predictions deviate from actual ratings by approximately 0.97 rating points.

## 📌Summary 
---
The notebook focuses on developing a personalized movie recommendation system using both collaborative and content-based filtering approaches

Key objectives include:

* Increasing user engagement and retention

* Improving customer satisfaction through personalization

* Enhancing revenue opportunities via targeted marketing

The methodology involves:

* Data cleaning and preprocessing of movie, rating, and link information

* Exploratory Data Analysis (EDA) of rating distributions and genre information

Implementation of two recommendation approaches:

* Collaborative Filtering: Based on user similarity patterns

* Content-Based Filtering: Based on movie content similarity

---

## 📌 Conclusion 
---
The analysis revealed several important findings:

* Rating distribution shows a positive skew, with most movies receiving 4-5 star ratings

* Both collaborative and content-based filtering methods demonstrated effectiveness in generating recommendations

* The combination of both approaches provides a more robust recommendation system

* The models show promise in capturing user preferences and suggesting relevant content

---
## 📌 Recommendation 
---
1. Hybrid System Implementation

  *   Combine collaborative and content-based filtering 

  *   Leverage the strengths of both methods for more accurate recommendations
Model Optimization

2. Implement continuous monitoring of model performance

  *   Regular updates to adapt to changing user preferences

  *   Consider implementing A/B testing for different recommendation strategies

3. Data Enhancement

  * Regular updates to adapt to changing user preferences

  * Expand the dataset with additional features:

    * User demographics

    * Movie reviews

    * Social media interactions

  * This will improve recommendation accuracy and personalization


4. User Engagement Strategy

  * Use personalized recommendations to increase platform engagement

  * Implement features to encourage content exploration

  * Track and analyze user interaction with recommendations

5. Technical Improvements

  * Regular system performance monitoring

  * Optimization of recommendation algorithms

  * Implementation of real-time recommendation updates

---

Contacts
        Github: https://github.com/MuiruriVivian/RECOMMENDER-SYSTEM---PHASE-4-PROJECT

