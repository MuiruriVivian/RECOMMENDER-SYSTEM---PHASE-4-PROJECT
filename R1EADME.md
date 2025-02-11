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

3. Hybrid Approach (Optional): Combine multiple recommendation techniques for better accuracy.

4. Performance Evaluation: Assess the effectiveness of different models using evaluation metrics such as RMSE (Root Mean Squared Error) and Cosine Similarity

## IMPORTING THE NECCESSRY LIBRARIES
---

*  Import necessary libraries for data handling, visualization, and modeling.

```python
!pip install surprise

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from surprise.prediction_algorithms import knns
from surprise import Reader, Dataset, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise.model_selection import cross_validate
```

    Collecting surprise
      Downloading surprise-0.1-py2.py3-none-any.whl.metadata (327 bytes)
    Collecting scikit-surprise (from surprise)
      Downloading scikit_surprise-1.1.4.tar.gz (154 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m154.4/154.4 kB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-surprise->surprise) (1.4.2)
    Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.11/dist-packages (from scikit-surprise->surprise) (1.26.4)
    Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from scikit-surprise->surprise) (1.13.1)
    Downloading surprise-0.1-py2.py3-none-any.whl (1.8 kB)
    Building wheels for collected packages: scikit-surprise
      Building wheel for scikit-surprise (pyproject.toml) ... [?25l[?25hdone
      Created wheel for scikit-surprise: filename=scikit_surprise-1.1.4-cp311-cp311-linux_x86_64.whl size=2505177 sha256=9abeccbb003fef5700ff3be7a649935b120682b2e84bea0ce50ba52793e34264
      Stored in directory: /root/.cache/pip/wheels/2a/8f/6e/7e2899163e2d85d8266daab4aa1cdabec7a6c56f83c015b5af
    Successfully built scikit-surprise
    Installing collected packages: scikit-surprise, surprise
    Successfully installed scikit-surprise-1.1.4 surprise-0.1


## LOADING THE DATASET
---
*   Read and explore the dataset.



```python
# upload links datasets
links = pd.read_csv("links.csv")
print(links.shape)
# links.head(5)

#upload movies dataset
movies = pd.read_csv("movies.csv")
print(movies.shape)
# movies.head(5)

# upload ratings dataset
ratings = pd.read_csv("ratings.csv")
print(ratings.shape)
#ratings.head(5)

```

    (9742, 3)
    (9742, 3)
    (100836, 4)


## DATA CLEANING

*   Merging the Datasets
*   Drop the irrelevant columns
*   Handle missing data.
*   Remove duplicate records, if any.
*   Clean or transform data types as necessary.









  <div id="df-80172460-9415-4cec-967e-d6e28ce8ec9f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>imdbId</th>
      <th>tmdbId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>114709</td>
      <td>862.0</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>114709</td>
      <td>862.0</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>5</td>
      <td>4.0</td>
      <td>847434962</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>114709</td>
      <td>862.0</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>7</td>
      <td>4.5</td>
      <td>1106635946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>114709</td>
      <td>862.0</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>15</td>
      <td>2.5</td>
      <td>1510577970</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>114709</td>
      <td>862.0</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>17</td>
      <td>4.5</td>
      <td>1305696483</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-80172460-9415-4cec-967e-d6e28ce8ec9f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-80172460-9415-4cec-967e-d6e28ce8ec9f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-80172460-9415-4cec-967e-d6e28ce8ec9f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-dd83704b-1a71-4ee9-b32b-d84487e41935">
  <button class="colab-df-quickchart" onclick="quickchart('df-dd83704b-1a71-4ee9-b32b-d84487e41935')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-dd83704b-1a71-4ee9-b32b-d84487e41935 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# drop columns
links_movies_ratings.drop(["imdbId", "tmdbId", "timestamp"], axis=1, inplace=True)
links_movies_ratings.head(5)
```





  <div id="df-def3bdc1-0dca-4d99-8d5b-1c2a7177ff9e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>7</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>15</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>17</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-def3bdc1-0dca-4d99-8d5b-1c2a7177ff9e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-def3bdc1-0dca-4d99-8d5b-1c2a7177ff9e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-def3bdc1-0dca-4d99-8d5b-1c2a7177ff9e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-0b810c3c-c900-4255-8c2e-05a25d5690f5">
  <button class="colab-df-quickchart" onclick="quickchart('df-0b810c3c-c900-4255-8c2e-05a25d5690f5')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-0b810c3c-c900-4255-8c2e-05a25d5690f5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# handle missing value
links_movies_ratings.dropna(inplace=True)
links_movies_ratings.isnull().sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>movieId</th>
      <td>0</td>
    </tr>
    <tr>
      <th>title</th>
      <td>0</td>
    </tr>
    <tr>
      <th>genres</th>
      <td>0</td>
    </tr>
    <tr>
      <th>userId</th>
      <td>0</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



## DATA PREPROSSESSING

> This is conducted in order to transform raw data into a structured format suitable for machine learning models.

*   Extracting year from movie titles.
*   Converting Data into Model-Specific Format
*   Splitting Data for Training & Testing







```python
# Extract year from movie title
links_movies_ratings['year'] = links_movies_ratings['title'].str.extract(r'\((\d{4})\)').astype(float)

# Remove the year from the title column
links_movies_ratings['title'] = links_movies_ratings['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)

# Display the updated dataframe
links_movies_ratings.head()

```





  <div id="df-40a32790-e9d2-49c2-b6ea-0865e3604c29" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>rating</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>1</td>
      <td>4.0</td>
      <td>1995.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>5</td>
      <td>4.0</td>
      <td>1995.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>7</td>
      <td>4.5</td>
      <td>1995.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>15</td>
      <td>2.5</td>
      <td>1995.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>[Adventure Animation Children Comedy Fantasy]</td>
      <td>17</td>
      <td>4.5</td>
      <td>1995.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-40a32790-e9d2-49c2-b6ea-0865e3604c29')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-40a32790-e9d2-49c2-b6ea-0865e3604c29 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-40a32790-e9d2-49c2-b6ea-0865e3604c29');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a4f48d6d-ed8b-45cc-86d4-80dda76b3bb4">
  <button class="colab-df-quickchart" onclick="quickchart('df-a4f48d6d-ed8b-45cc-86d4-80dda76b3bb4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a4f48d6d-ed8b-45cc-86d4-80dda76b3bb4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Pipeline


```python
# Data Preprocessing Pipeline
def preprocess_movies(movies_df):
    links_movies_ratings['genres'] = links_movies_ratings['genres'].str.replace('|', ' ')
    vectorizer = TfidfVectorizer(stop_words='english')
    genre_matrix = vectorizer.fit_transform(links_movies_ratings['genres'])
    return genre_matrix

movies_tfidf_matrix = preprocess_movies(movies)
print(movies_tfidf_matrix.shape)
print(" ")
print("There are 9742 movies in the dataset.")
print("There are 23 unique genre-related words in the dataset (after processing).")
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-90-653a5c4a30cc> in <cell line: 0>()
          6     return genre_matrix
          7 
    ----> 8 movies_tfidf_matrix = preprocess_movies(movies)
          9 print(movies_tfidf_matrix.shape)
         10 print(" ")


    <ipython-input-90-653a5c4a30cc> in preprocess_movies(movies_df)
          3     links_movies_ratings['genres'] = links_movies_ratings['genres'].str.replace('|', ' ')
          4     vectorizer = TfidfVectorizer(stop_words='english')
    ----> 5     genre_matrix = vectorizer.fit_transform(links_movies_ratings['genres'])
          6     return genre_matrix
          7 


    /usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
       2102             sublinear_tf=self.sublinear_tf,
       2103         )
    -> 2104         X = super().fit_transform(raw_documents)
       2105         self._tfidf.fit(X)
       2106         # X is already a transformed view of raw_documents so


    /usr/local/lib/python3.11/dist-packages/sklearn/base.py in wrapper(estimator, *args, **kwargs)
       1387                 )
       1388             ):
    -> 1389                 return fit_method(estimator, *args, **kwargs)
       1390 
       1391         return wrapper


    /usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
       1374                     break
       1375 
    -> 1376         vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
       1377 
       1378         if self.binary:


    /usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _count_vocab(self, raw_documents, fixed_vocab)
       1261         for doc in raw_documents:
       1262             feature_counter = {}
    -> 1263             for feature in analyze(doc):
       1264                 try:
       1265                     feature_idx = vocabulary[feature]


    /usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _analyze(doc, analyzer, tokenizer, ngrams, preprocessor, decoder, stop_words)
        102     else:
        103         if preprocessor is not None:
    --> 104             doc = preprocessor(doc)
        105         if tokenizer is not None:
        106             doc = tokenizer(doc)


    /usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _preprocess(doc, accent_function, lower)
         60     """
         61     if lower:
    ---> 62         doc = doc.lower()
         63     if accent_function is not None:
         64         doc = accent_function(doc)


    AttributeError: 'float' object has no attribute 'lower'


Split and Train the Data


```python
# Split the dataset into training and testing set
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
```

## EXPLORATORY DATA ANALYSIS (EDA)

*   Basic statistics.
*   Data visualization.



###  1.Rating


```python
# EDA for rating: plot histogram
plt.figure(figsize=(8, 6))
sns.histplot(links_movies_ratings['rating'], bins=20, kde=True)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```


    
![png](output_18_0.png)
    


**EXPLANATION**

> The Histogram shows that majority of the movies are rated 4 while very few had a rating of 1.


> The distribution is skewed towards higher ratings, with the highest frequency at Rating 4.

> As the ratings increase from 1 to 5, the number of counts increases too, showing that more movies received higher ratings.


> This pattern suggests that, in this dataset, movies are more likely to receive higher ratings than lower ones.














### 2.Genre:


```python
# Each movie can have multiple genres separated by '|'. We need to split them.

# Create a copy of the dataframe to explode genres
df_genre = links_movies_ratings.copy()
# Replace '(no genres listed)' with NaN maybe
# Split genres by '|'
df_genre['genres'] = df_genre['genres'].fillna('')
df_genre = df_genre.assign(genre = df_genre['genres'].str.split('\|'))


# Explode to have one genre per row
df_exploded = df_genre.explode('genre')

# Count per genre overall
genre_counts = df_exploded['genre'].value_counts().reset_index()
genre_counts.columns = ['genre', 'count']

# Plot genre counts
plt.figure(figsize=(15, 8))
sns.barplot(data=genre_counts, x='genre', y='count')
plt.title('Counts of Movies by Genre')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=55)
plt.show()
print(" ")
print("This means Drama is the most common genre in the dataset, followed by Comedy and Action")
```


    
![png](output_21_0.png)
    


     
    This means Drama is the most common genre in the dataset, followed by Comedy and Action


**EXPLANATION**

> Drama is the most frequent genre, with over 40,000 movies. It is followed closely by Comedy and Action, which also have large counts. Genres such as Thriller, Adventure, and Romance appear in the middle range, with counts significantly lower than Drama but still in the high teens to low 20,000s



> On the other end of the spectrum, genres like Western, Documentary, and Film-Noir are much less frequent, with counts well under 5,000.




### 3.Rating by Title


```python
# Rating group by title
avg_rating_title = links_movies_ratings.groupby('title')['rating'].mean().reset_index()
avg_rating_title = avg_rating_title.sort_values('rating', ascending=False)
print('Average rating per title:')
print(avg_rating_title.head(10))
print(" ")
print("The top 5 Most rated movies are Hollywood Chainsaw Hookers, Calcium Kid,Chinese Puzzle (Casse-tête chinois), Raise Your Voice and Rain")
print(" ")
print(avg_rating_title.tail(10))
print("The Least rated movies are Indestructible Man,Yongary: Monster from the Deep,  Don't Look Now, Superfast! and Anaconda: The Offspring")
```

    Average rating per title:
                                               title  rating
    3863                  Hollywood Chainsaw Hookers     5.0
    1473                            Calcium Kid, The     5.0
    1692         Chinese Puzzle (Casse-tête chinois)     5.0
    6742                            Raise Your Voice     5.0
    6738                                        Rain     5.0
    6727                                   Radio Day     5.0
    8463                          Thousand Clowns, A     5.0
    4013                           Hunting Elephants     5.0
    1183                              Blue Planet II     5.0
    760   Ballad of Narayama, The (Narayama bushiko)     5.0
     
    The top 5 Most rated movies are Hollywood Chainsaw Hookers, Calcium Kid,Chinese Puzzle (Casse-tête chinois), Raise Your Voice and Rain
     
                                       title  rating
    4865              Leprechaun 4: In Space     0.5
    7515                             Skyline     0.5
    7099  Rust and Bone (De rouille et d'os)     0.5
    7965                            Survivor     0.5
    4953                           Lionheart     0.5
    457              Anaconda: The Offspring     0.5
    7945                          Superfast!     0.5
    2414                      Don't Look Now     0.5
    9369      Yongary: Monster from the Deep     0.5
    4212                  Indestructible Man     0.5
    The Least rated movies are Indestructible Man,Yongary: Monster from the Deep,  Don't Look Now, Superfast! and Anaconda: The Offspring


### 4.Rating per Genre


```python
# calculate average rating per genre
avg_rating_genre = df_exploded.groupby('genre')['rating'].mean().reset_index()
avg_rating_genre = avg_rating_genre.sort_values('rating', ascending=False)
print('Average rating per genre:')
print(avg_rating_genre)
```

    Average rating per genre:
                     genre    rating
    10           Film-Noir  3.920115
    18                 War  3.808294
    7          Documentary  3.797785
    6                Crime  3.658294
    8                Drama  3.656184
    14             Mystery  3.632460
    3            Animation  3.629937
    12                IMAX  3.618335
    19             Western  3.583938
    13             Musical  3.563678
    2            Adventure  3.508609
    15             Romance  3.506511
    17            Thriller  3.493706
    9              Fantasy  3.491001
    0   (no genres listed)  3.489362
    16              Sci-Fi  3.455721
    1               Action  3.447984
    4             Children  3.412956
    5               Comedy  3.384721
    11              Horror  3.258195


## BUILD THE RECOMMENDER SYSTEM


### A. Collaborative Filtering Using Surprise


```python
# Converting Data into Model-Specific Format
# For Surprise Library (Collaborative Filtering Model):

# Define the rating scale
reader = Reader(rating_scale=(0.5, 5.0))

# Load data into Surprise Dataset
data = Dataset.load_from_df(links_movies_ratings[['userId', 'movieId', 'rating']], reader)

# Load dataset into Surprise format
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(links_movies_ratings[['userId', 'movieId', 'rating']], reader)

# Train model using SVD
model = SVD()
# Import cross_validate from surprise.model_selection

cross_validate(model, data, cv=5, verbose=True)
```

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
    
                      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.8748  0.8701  0.8726  0.8760  0.8713  0.8729  0.0022  
    MAE (testset)     0.6715  0.6650  0.6728  0.6723  0.6699  0.6703  0.0028  
    Fit time          1.56    1.59    1.59    1.57    1.57    1.57    0.01    
    Test time         0.10    0.10    0.10    0.10    0.35    0.15    0.10    





    {'test_rmse': array([0.87477704, 0.87005596, 0.87260726, 0.87595396, 0.87131472]),
     'test_mae': array([0.67153269, 0.66500372, 0.67275003, 0.67231687, 0.66986281]),
     'fit_time': (1.5587129592895508,
      1.5860569477081299,
      1.5892562866210938,
      1.5710761547088623,
      1.5672359466552734),
     'test_time': (0.10378479957580566,
      0.10296154022216797,
      0.10126829147338867,
      0.10281133651733398,
      0.346189022064209)}




```python
# Train a basic SVD model on the training set
algo = SVD(random_state=42)
algo.fit(trainset)

# Predict on the testset
predictions = algo.test(testset)

# Compute and print RMSE
rmse = accuracy.rmse(predictions)

print('RMSE on test set:', rmse)
print('Number of predictions:', len(predictions))
```

    RMSE: 0.8745
    RMSE on test set: 0.8744981021934208
    Number of predictions: 25209


### B. Content-Based Filtering (Using Cosine Similarity)


```python
# Convert movie genres into a TF-IDF matrix
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(""))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend(movie_title, num_recommendations=5):
    idx = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended_movie_indices = [i[0] for i in scores[1:num_recommendations+1]]
    return movies.iloc[recommended_movie_indices]['title']

# Example recommendation
recommend("Toy Story (1995)")

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-93-a6c79dd31d18> in <cell line: 0>()
          1 # Convert movie genres into a TF-IDF matrix
          2 tfidf = TfidfVectorizer(stop_words="english")
    ----> 3 tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(""))
          4 
          5 # Compute cosine similarity


    /usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
       2102             sublinear_tf=self.sublinear_tf,
       2103         )
    -> 2104         X = super().fit_transform(raw_documents)
       2105         self._tfidf.fit(X)
       2106         # X is already a transformed view of raw_documents so


    /usr/local/lib/python3.11/dist-packages/sklearn/base.py in wrapper(estimator, *args, **kwargs)
       1387                 )
       1388             ):
    -> 1389                 return fit_method(estimator, *args, **kwargs)
       1390 
       1391         return wrapper


    /usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
       1374                     break
       1375 
    -> 1376         vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
       1377 
       1378         if self.binary:


    /usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _count_vocab(self, raw_documents, fixed_vocab)
       1280             vocabulary = dict(vocabulary)
       1281             if not vocabulary:
    -> 1282                 raise ValueError(
       1283                     "empty vocabulary; perhaps the documents only contain stop words"
       1284                 )


    ValueError: empty vocabulary; perhaps the documents only contain stop words


## MODEL EVALUATION


```python
# Define similarity options
sim_options = {
    'name': 'cosine',  # Use cosine similarity to measure the similarity between items
    'user_based': False  # Set to False for item-based filtering (True would be for user-based filtering)
}

# Build the model using the KNNBasic algorithm
item_cf_model = KNNBasic(sim_options=sim_options)

# Train the model on the training set
item_cf_model.fit(trainset)
```

    Computing the cosine similarity matrix...
    Done computing similarity matrix.





    <surprise.prediction_algorithms.knns.KNNBasic at 0x7c4873d0ff10>




```python
trainset, testset = train_test_split(data, test_size=0.2)
model.fit(trainset)
predictions = model.test(testset)

# Compute RMSE
rmse = accuracy.rmse(predictions)
print('RMSE:', rmse)

```

    RMSE: 0.8639
    RMSE: 0.8638545783772473


## MAKING A PREDICTION


```python
sim_cosine = {"name": "cosine", "user_based": False}
basic_cosine = knns.KNNBasic(sim_options=sim_cosine)
basic_cosine.fit(trainset)
predictions = basic_cosine.test(testset)
print(accuracy.rmse(predictions))
```

    Computing the cosine similarity matrix...
    Done computing similarity matrix.
    RMSE: 0.9717
    0.9716548174439473



Predict ratings for a user-movie pair.


```python
user_id = 1
movie_id = 9745
predicted_rating = model.predict(user_id, movie_id)
print(f"Predicted rating: {predicted_rating.est}")
```

    Predicted rating: 4.187465617028602



```python
user_id = 1
movie_id = 50
predicted_rating = model.predict(user_id, movie_id)
print(f"Predicted rating: {predicted_rating.est}")

```

    Predicted rating: 5.0



```python
# Define movie titles
movie_title_1 = "Avenger"
movie_title_2 = "Ex Drummer"

# Find the movie IDs for the given titles from the combined DataFrame
movie_id_1 =links_movies_ratings[links_movies_ratings['title'].str.contains(movie_title_1, case=False, na=False)]['movieId'].values
movie_id_2 = links_movies_ratings[links_movies_ratings['title'].str.contains(movie_title_2, case=False, na=False)]['movieId'].values

# Check if the movie titles were found
if len(movie_id_1) > 0:
    rating_1 = links_movies_ratings[links_movies_ratings['movieId'] == movie_id_1[0]]['rating'].values[0]
    print(f"Movie: {movie_title_1}, Rating: {rating_1}")
else:
    print(f"Movie '{movie_title_1}' not found.")

if len(movie_id_2) > 0:
    rating_2 = links_movies_ratings[links_movies_ratings['movieId'] == movie_id_2[0]]['rating'].values[0]
    print(f"Movie: {movie_title_2}, Rating: {rating_2}")
else:
    print(f"Movie '{movie_title_2}' not found.")

```

    Movie: Avenger, Rating: 3.0
    Movie: Ex Drummer, Rating: 5.0


## MAKE RECOMMENDATION BASED ON A MOVIE TITLE


```python
def get_similar_movies(movie_title, model, trainset, movies_df, top_n=5):
    # Find the movie ID for the given title
    movie_id = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]['movieId'].values

    # Convert the movieId to an internal ID used by Surprise (trainset)
    movie_inner_id = trainset.to_inner_iid(movie_id[0])

    # Get the top N most similar movies using the KNN model's get_neighbors function
    neighbors = model.get_neighbors(movie_inner_id, k=top_n)

    # Map internal IDs back to movie titles
    similar_titles = [(movies_df[movies_df['movieId'] == int(trainset.to_raw_iid(neighbor))]['title'].values[0])
                      for neighbor in neighbors]

    return similar_titles

# Example usage
recommend("Toy Story", 5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1706</th>
      <td>Antz (1998)</td>
    </tr>
    <tr>
      <th>2355</th>
      <td>Toy Story 2 (1999)</td>
    </tr>
    <tr>
      <th>2809</th>
      <td>Adventures of Rocky and Bullwinkle, The (2000)</td>
    </tr>
    <tr>
      <th>3000</th>
      <td>Emperor's New Groove, The (2000)</td>
    </tr>
    <tr>
      <th>3568</th>
      <td>Monsters, Inc. (2001)</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>




```python
# Movie title input
movie_title = "Shooter"

# Get the top 5 similar movies
recommended_movies = get_similar_movies(movie_title, item_cf_model, trainset, links_movies_ratings, top_n=5)

# Print the recommended movies (only titles)
if isinstance(recommended_movies, list):
    print(f"Top 5 similar movies to '{movie_title}':")
    for movie in recommended_movies:
        print(movie)
else:
    print(recommended_movies)
```

    Top 5 similar movies to 'Shooter':
    Sum of All Fears, The (2002)
    Working Girl (1988)
    A Million Ways to Die in the West (2014)
    Enough (2002)
    Mothman Prophecies, The (2002)



```python
# Movie title input
movie_title = "Hollywood Chainsaw Hookers"

# Get the top 5 similar movies
recommended_movies = get_similar_movies(movie_title, item_cf_model, trainset, links_movies_ratings, top_n=5)

# Print the recommended movies (only titles)
if isinstance(recommended_movies, list):
    print(f"Top 5 similar movies to '{movie_title}':")
    for movie in recommended_movies:
        print(movie)
else:
    print(recommended_movies)

```

    Top 5 similar movies to 'Hollywood Chainsaw Hookers':
    Ferris Bueller's Day Off (1986)
    My Big Fat Greek Wedding (2002)
    101 Dalmatians (One Hundred and One Dalmatians) (1961)
    Meet the Parents (2000)
    American President, The (1995)


## MAKE RECOMMENDATION BASED ON GENRE

Explode Genres and Count Views per Genre


```python
# Split genres into separate rows
movies['genres'] = movies['genres'].str.split('|')
movies_exploded = movies.explode('genres')

# Merge ratings with movies dataset
user_genre_data = ratings.merge(movies_exploded, on="movieId")

# Count number of times each user has watched a genre
user_genre_counts = user_genre_data.groupby(['userId', 'genres']).size().reset_index(name="count")

# Display the first few rows
print("This table shows how many times each user watched a specific genre.")
print(" ")
user_genre_counts.head()
```

    This table shows how many times each user watched a specific genre.
     






  <div id="df-25a29b45-2198-418e-acba-c8cede6f3567" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>genres</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Action Adventure</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Action Adventure Children Comedy Fantasy</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Action Adventure Children Fantasy Mystery Thri...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Action Adventure Comedy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Action Adventure Comedy Fantasy</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-25a29b45-2198-418e-acba-c8cede6f3567')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-25a29b45-2198-418e-acba-c8cede6f3567 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-25a29b45-2198-418e-acba-c8cede6f3567');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-fa815ae1-e1af-495c-a2ae-40139d59e193">
  <button class="colab-df-quickchart" onclick="quickchart('df-fa815ae1-e1af-495c-a2ae-40139d59e193')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-fa815ae1-e1af-495c-a2ae-40139d59e193 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Find the most-watched genre for each user
favorite_genres = user_genre_counts.loc[user_genre_counts.groupby('userId')['count'].idxmax()]

# Display a few users and their favorite genres
print("This finds the genre each user watches the most")
print(" ")
favorite_genres.head()
```

    This finds the genre each user watches the most
     






  <div id="df-f42d97c1-4ffd-4abf-9e91-2665bca2cf66" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>genres</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>Action Adventure Sci-Fi</td>
      <td>11</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2</td>
      <td>Comedy</td>
      <td>3</td>
    </tr>
    <tr>
      <th>168</th>
      <td>3</td>
      <td>Drama</td>
      <td>4</td>
    </tr>
    <tr>
      <th>211</th>
      <td>4</td>
      <td>Comedy</td>
      <td>21</td>
    </tr>
    <tr>
      <th>287</th>
      <td>5</td>
      <td>Crime Drama</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f42d97c1-4ffd-4abf-9e91-2665bca2cf66')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f42d97c1-4ffd-4abf-9e91-2665bca2cf66 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f42d97c1-4ffd-4abf-9e91-2665bca2cf66');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-cb94613b-2110-4a00-80b3-8e68ad4d8c3a">
  <button class="colab-df-quickchart" onclick="quickchart('df-cb94613b-2110-4a00-80b3-8e68ad4d8c3a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-cb94613b-2110-4a00-80b3-8e68ad4d8c3a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
def recommend_by_genre(user_id, num_recommendations=5):
    # Get user's favorite genre
    fav_genre = favorite_genres.loc[favorite_genres['userId'] == user_id, 'genres'].values[0]

    # Find movies that belong to this genre
    recommended_movies = movies_exploded[movies_exploded['genres'] == fav_genre]

    # Sort by average rating (or another metric)
    # Ensure 'movieId' is treated as numeric before merging
    recommended_movies['movieId'] = pd.to_numeric(recommended_movies['movieId'])
    top_movies = recommended_movies.merge(ratings, on='movieId').groupby(['movieId', 'title'])['rating'].mean().reset_index()

    # Sort and get top recommendations
    top_movies = top_movies.sort_values(by='rating', ascending=False)

    return top_movies[['title', 'rating']].head(num_recommendations)

# Example usage:
print("This function finds movies that match the user's preferred genre and recommends the top-rated ones.")
print(" ")
recommend_by_genre(1, 5)  # Recommend 5 movies for user 1
```

    This function finds movies that match the user's preferred genre and recommends the top-rated ones.
     






  <div id="df-92023404-4bff-4445-8885-b4d6cd75d104" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Star Wars: Episode IV - A New Hope (1977)</td>
      <td>4.231076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Star Wars: Episode V - The Empire Strikes Back...</td>
      <td>4.215640</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Star Wars: Episode VI - Return of the Jedi (1983)</td>
      <td>4.137755</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Guardians of the Galaxy (2014)</td>
      <td>4.050847</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Thor: Ragnarok (2017)</td>
      <td>4.025000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-92023404-4bff-4445-8885-b4d6cd75d104')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-92023404-4bff-4445-8885-b4d6cd75d104 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-92023404-4bff-4445-8885-b4d6cd75d104');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a9b23c26-feb4-4729-8f93-d3b45e61c52b">
  <button class="colab-df-quickchart" onclick="quickchart('df-a9b23c26-feb4-4729-8f93-d3b45e61c52b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a9b23c26-feb4-4729-8f93-d3b45e61c52b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>



