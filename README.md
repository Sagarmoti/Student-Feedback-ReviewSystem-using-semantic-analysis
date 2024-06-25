# %% [markdown]
# # Sentiment Analysis on Student Feedback in Engineering Education

# %% [markdown]
# ![sentiment2.jpg](sentiment2.jpg)

# %% [markdown]
# ## Introduction
# 
# In the field of engineering education, student feedback plays a vital role in assessing the effectiveness of teaching methods, course materials, and overall learning experiences. Sentiment analysis, a key component of data science, offers a powerful approach to analyze and extract valuable insights from student feedback. <br>
# 
# The objective of this project is to perform sentiment analysis on the feedback provided by  computer engineering students in the University of GITM (my course mates). By making use of natural language processing (NLP) techniques and machine learning algorithms, we aim to uncover sentiments expressed in the feedback and gain a comprehensive understanding of student perceptions, satisfaction, and areas of improvement.<br>
# 
# Through the analysis of student feedback, we can identify common themes, sentiment trends, and specific challenges faced by students. This valuable information can help inform the department and it's lectureres about the effectiveness of their teaching methodologies, course content, and student support systems. The insights derived from sentiment analysis on student feedback can drive evidence-based decision-making in engineering education. It enables the department to address concerns, make improvements, and create a positive learning environment that caters to the needs of the students.

# %% [markdown]
# ## Data Collection
# To get the data to use for this project, I utilized [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLScRYfjQNUb_kY4dh3wJasOQucJO2YcdT6xXaSyCRlwYz3OXng/viewform) to collect valuable feedback from students. The platform facilitated the collection of diverse responses, streamlined data collection, ensuring accuracy and efficiency in gathering student sentiments and  provided a comprehensive dataset for the sentiment analysis.

# %%
# import libraries and packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='white')

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

# %%
# colors

my_colors = ["#c6b34e","#95354a", "#57b9a8","#b0ddff", "#426872","#215c97", "#02b6b4","#b05468","#cd9f62","#aaaaaa","#8fce00","#827861"]
# Create a seaborn palette object
my_palette = sns.color_palette(my_colors)
# Use the custom palette with a seaborn function
sns.set_palette(my_palette)
from matplotlib.colors import ListedColormap
colors = ['#ffffcd', '#ffeaa4', '#ffca2a','#c6b34e']
my_cmap = sns.color_palette(colors)
cmap = ListedColormap(colors)

# %%
# load the dataset and show first 5 rows
df = pd.read_csv('Sentiment Analysis on Student Feedback.csv')
df.head()

# %%
df.info()

# %% [markdown]
# ## Data Cleaning
# Here, I'm going to clean the dataset as it can be seen to have some quality issues.

# %%
df.columns

# %%
df.columns = df.columns.str.strip()
print(df.columns)

# %%
# drop unncessary column
df = df.drop(['Unnamed: 11'], axis=1)

# Convert the column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract the date and time into separate columns
df['Date'] = df['Timestamp'].dt.date
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = df['Timestamp'].dt.time
df['Hour'] = df['Timestamp'].dt.hour

# drop Timestamp column
df = df.drop(['Timestamp'], axis=1)

# corrections to "Study Hours (per week) column"
df['Study Hours (per week)'] = df['Study Hours (per week)'].str.extract(r'(\d+)').fillna(0).astype(int)

# %%
# overview of the data again
df.info()

# %%
# preview random sample of the data
df.sample(5)

# %% [markdown]
# ## Data Preprocessing
# Cleaning and preprocessing the data by handling contractions, converting text to lower case removing stop words, punctuations, hashtags, numbers/digits and special characters and then tokenizing and lemmatizing the text.

# %%
def preprocess_text(text):
    """
    Preprocess a text string for sentiment analysis.

    Parameters
    ----------
    text : str
        The text string to preprocess.

    Returns
    -------
    str
        The preprocessed text string.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs, hashtags, mentions, and special characters
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers/digits
    text = re.sub(r'\b[0-9]+\b\s*', '', text)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a single string
    return ' '.join(tokens)

df['Processed_Feedback'] = df['Feedback'].apply(preprocess_text)
df.tail(10)

# %%
# lets check out the new processed text column
df['Processed_Feedback'][:10].to_frame()

# %% [markdown]
# ## Language Detection
# Detecting the type of language used in the feedback text.

# %%
from langdetect import detect

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return None
    
df['Language'] = df['Processed_Feedback'].apply(detect_language)

# %%
df['Language'].unique()

# %%
language_mapping = {
    'en': 'English',
    'cy': 'Welsh',
    'so': 'Somali',
    'sk': 'Slovak',
    'af': 'Afrikaans',
    'fr': 'French',
    'hr': 'Croatian',
    'id': 'Indonesian',
    'pt': 'Portuguese',
    'it': 'Italian',
    'pl': 'Polish',
    'es': 'Spanish'
}

df['Language'] = df['Language'].map(language_mapping)
df['Language'].unique()

# %%
df

# %% [markdown]
# The language is not important in this analysis though as it is not consistent with the feedback text. So I'll drop the column.

# %% [markdown]
# ### Feature Engineering

# %%
df['Char_Count'] = df['Processed_Feedback'].apply(len) # can also use df['Processed_Feedback'].str.len()
df['Word_Count'] = df['Processed_Feedback'].apply(lambda x: len(x.split()))
df = df.drop(['Language'], axis=1)

# %%
df.sort_values(by='Char_Count', ascending=False).head(10)

# %% [markdown]
# ## Sentiment scores and Labels
# Calaculating the sentiment scores and it's corresponding labels.

# %%
# Calculate sentiment scores
df['Sentiment_Scores'] = df['Processed_Feedback'].apply(lambda x: TextBlob(x).sentiment.polarity)

print(df['Sentiment_Scores'].describe())

sns.histplot(df['Sentiment_Scores'])
plt.title('Distribution of sentiment scores')
plt.ylabel('Frequency')
plt.xlabel('Sentiment scores')
plt.show()

# %%
def sentiment_analyzer(score):
    """
    Classify sentiment based on a score.

    Parameters
    ----------
    score : float
        The sentiment score to classify.

    Returns
    -------
    str
        The sentiment label.
    """

    if score > 0.049889:
        return 'positive'
    
    elif score < 0.049889:
        return 'negative'
    
    else:
        return 'neutral'


# Assuming you have a DataFrame named 'df' with a column 'Sentiment_Scores'
df['Sentiments'] = df['Sentiment_Scores'].apply(sentiment_analyzer)

# %%
print(list(df['Sentiment_Scores']))

df['Sentiments'].value_counts()

# %%
# # Calculate subjectivity scores
# df['Subjectivity_Score'] = df['Processed_Feedback'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# # Map sentiment scores to sentiment labels
# df['Sentiments'] = df.apply(lambda row: 'Positive' if row['Sentiment_Scores'] > 0.03 and row['Subjectivity_Score'] > 0.5 else 
#                                         'Negative' if row['Sentiment_Scores'] < -0.03 and row['Subjectivity_Score'] < 0.5 else 'Neutral', axis=1)
# df = df.drop(['Subjectivity_Score'], axis=1)

# %% [markdown]
# ## Aspect-Based sentiment Analysis Metrics

# %% [markdown]
# ### Summary statistics and metrics 

# %%
# Sentiment Analysis Metrics
sentiment_counts = df['Sentiments'].value_counts()

# Descriptive Statistics
study_hours_stats = df['Study Hours (per week)'].describe()
overall_satisfaction_stats = df['Overall Satisfaction'].describe()

# Categorical Metrics
course_code_counts = df['Course Code'].value_counts()
department_counts = df['Department'].value_counts()
sentiment_distribution = df.groupby('Course Code')['Sentiments'].value_counts(normalize=True)

# Print the calculated metrics
print("Sentiment Analysis Metrics:")
print(sentiment_counts)
print("\nDescriptive Statistics - Study Hours:")
print(study_hours_stats)
print("\nDescriptive Statistics - Overall Satisfaction:")
print(overall_satisfaction_stats)
print("\nCategorical Metrics - Course Code Counts:")
print(course_code_counts)
print("\nCategorical Metrics - Department Counts:")
print(department_counts)
print("\nSentiment Distribution by Course Code:")
print(sentiment_distribution)

# %% [markdown]
# ### Analyzing the frequency of specific keywords or phrases in the feedback

# %%
# analyze the frequency of specific keywords or phrases in the feedback
from collections import Counter

# The keywords or phrases of interest
keywords = ['shit', 'difficult', 'terrible', 'okay', 'best', 'worst', 'good', 'try', 'bad']

# Concatenate all the preprocessed feedback into a single string
all_feedback = ' '.join(df['Processed_Feedback'])

# Tokenize the text into individual words
tokens = all_feedback.split()

# Count the frequency of each keyword in the feedback
keyword_frequency = Counter(tokens)

# Print the frequency of each keyword
for keyword in keywords:
    print(f"Frequency of '{keyword}': {keyword_frequency[keyword]}")

# %%
# analyze the frequency of specific keywords or phrases in the feedback

negative_keywords = [
    'shit',
    'difficult',
    'terrible',
    'okay',
    'worst',
    'bad',
    'try',
    'boring',
    'confusing',
    'disappointed',
    'discouraged',
    'dumb',
    'frustrated',
    'horrible',
    'lame',
    'lousy',
    'miserable',
    'pointless',
    'stupid',
    'useless'
]

positive_keywords = [
    'amazing',
    'awesome',
    'brilliant',
    'clear',
    'clever',
    'creative',
    'helpful',
    'inspiring',
    'interesting',
    'intelligent',
    'lovely',
    'nice',
    'outstanding',
    'perfect',
    'wonderful'
]

# Concatenate all the feedback into a single string
all_feedback = ' '.join(df['Processed_Feedback'])

# Tokenize the text into individual words
tokens = nltk.word_tokenize(all_feedback)

# Filter out stop words
stop_words = nltk.corpus.stopwords.words('english')
filtered_tokens = [token for token in tokens if token not in stop_words]

# Create a dictionary to store the frequency of each keyword
keyword_frequency = Counter(filtered_tokens)

# Print the frequency of each keyword
for keyword in negative_keywords:
    print(f"Frequency of '{keyword}': {keyword_frequency[keyword]}")

for keyword in positive_keywords:
    print(f"Frequency of '{keyword}': {keyword_frequency[keyword]}")

# %% [markdown]
# ## Topic Modeling
# Implementing the topic modeling technique, **Latent Dirichlet Allocation (LDA)** to identify underlying topics or themes in the feedback data. This can provide deeper insights into the content and help analyze sentiment within specific topics.

# %%
# Create a CountVectorizer
vectorizer = CountVectorizer(max_features=1000, lowercase=True, stop_words='english', ngram_range=(1, 2))

# Apply CountVectorizer to the processed feedback text
dtm = vectorizer.fit_transform(df['Processed_Feedback'])

# Perform LDA topic modeling
num_topics = 10  # Specify the desired number of topics
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(dtm)

# Get the top words for each topic
feature_names = vectorizer.get_feature_names_out()
top_words = 10  # Specify the number of top words to retrieve for each topic
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]]))
    print()

# %% [markdown]
# ## Exploratory Data Analysis
# Creating meaningful visualizations to gain insights and communicate findings effectively. Exploring different types of plots, charts, and graphs to showcase various aspects of the data and also analyzing the distribution of sentiment labels in the data to understand the overall sentiment polarity.

# %% [markdown]
# ### Correlation Analysis
# Exploring the correlation between sentiment and other variables in the dataset to identify potential relationships.

# %%
correlation_matrix = df[['Study Hours (per week)', 'Overall Satisfaction']].corr()

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation between Study Hours and Overall Satisfaction')

for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        if i != j:
            text = '{:.2f}'.format(correlation_matrix.iloc[i, j])
            plt.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black')

colorbar = plt.gca().collections[0].colorbar
colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
colorbar.set_ticklabels(['Strong Negative', 'Negative', 'Neutral', 'Positive', 'Strong Positive'])

plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

# %%
correlation_matrix = df[['Sentiment_Scores', 'Overall Satisfaction']].corr()

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation between Sentiment Score and Satisfaction')

for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        if i != j:
            text = '{:.2f}'.format(correlation_matrix.iloc[i, j])
            plt.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black')

colorbar = plt.gca().collections[0].colorbar
colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
colorbar.set_ticklabels(['Strong Negative', 'Negative', 'Neutral', 'Positive', 'Strong Positive'])

plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

# %%
# Bar plot for Course Code
plt.figure(figsize=(10, 6))
order = df['Course Code'].value_counts().index
ax = sns.countplot(data=df, x='Course Code', order=order)
plt.xlabel('Course Code')
plt.ylabel('Count of Feedback')
plt.title('Feedback Count by Course Code')
plt.xticks(rotation=45)
ax.bar_label(ax.containers[0], fmt='%.0f', label_type='edge')
plt.show()

# %%
# Word cloud for Overall Feedback: Combine all feedback into a single string
all_feedback = ' '.join(df['Processed_Feedback'])

plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_feedback)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Overall Feedback')
plt.show()

# %%
# Word cloud for Positive Feedback
data = ' '.join(df[df['Processed_Feedback'] == 'positive']['Processed_Feedback'])

plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_feedback)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Positive Feedback')
plt.show()

# %%
# Word cloud for Negative Feedback
data = ' '.join(df[df['Processed_Feedback'] == 'negative']['Processed_Feedback'])

plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_feedback)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Negative Feedback')
plt.show()

# %%
# Word cloud for Neutral Feedback
data = ' '.join(df[df['Processed_Feedback'] == 'neutral']['Processed_Feedback'])

plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_feedback)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Neutral Feedback')
plt.show()

# %%
# Bar plot for Sentiment
plt.figure(figsize=(8, 6))
order = df['Sentiments'].value_counts().index
ax = sns.countplot(data=df, x='Sentiments', order=order)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Sentiments')
plt.show()

# %%
label_data = df['Sentiments'].value_counts()
# explode = (0.1, 0.1, 0.1)
explode = (0.1, 0.1)
plt.figure(figsize=(8, 6))
patches, texts, pcts = plt.pie(label_data,labels = label_data.index,pctdistance = 0.65,shadow = True,startangle = 90,explode = explode,
                               autopct = '%1.1f%%',textprops={ 'fontsize': 10,'color': 'black','weight': 'bold','family': 'serif' })
plt.setp(pcts, color='white')
hfont = {'fontname':'serif', 'weight': 'bold'}
plt.title('Label', size=10, **hfont)
centre_circle = plt.Circle((0,0),0.40,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

# %%
# Bar plot for Previous Experience
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df, x='Previous Experience')
plt.xlabel('Previous Experience')
plt.ylabel('Count')
plt.title('Feedback Count by Previous Experience')
ax.bar_label(ax.containers[0], fmt='%.0f', label_type='edge')
plt.show()

# %%
label_data = df['Gender'].value_counts()
explode = (0.1, 0.1)
plt.figure(figsize=(8, 6))
patches, texts, pcts = plt.pie(label_data,labels = label_data.index,pctdistance = 0.65,shadow = True,startangle = 90,explode = explode,
                               autopct = '%1.1f%%',textprops={ 'fontsize': 10,'color': 'black','weight': 'bold','family': 'serif' })
plt.setp(pcts, color='white')
hfont = {'fontname':'serif', 'weight': 'bold'}
plt.title('Label', size=10, **hfont)
centre_circle = plt.Circle((0,0),0.40,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

# %%
label_data = df['Attendance'].value_counts()
explode = (0.1, 0.1, 0.1)
plt.figure(figsize=(8, 6))
patches, texts, pcts = plt.pie(label_data,labels = label_data.index,pctdistance = 0.65,shadow = True,startangle = 90,explode = explode,
                               autopct = '%1.1f%%',textprops={ 'fontsize': 10,'color': 'black','weight': 'bold','family': 'serif' })
plt.setp(pcts, color='white')
hfont = {'fontname':'serif', 'weight': 'bold'}
plt.title('Label', size=10, **hfont)
centre_circle = plt.Circle((0,0),0.40,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

# %%
# Bar plot for Course Difficulty
plt.figure(figsize=(10, 6))
order = ['Easy', 'Moderate', 'Challenging', 'Difficult']
ax = sns.countplot(data=df, x='Course Difficulty', order=order)
plt.xlabel('Course Difficulty')
plt.ylabel('Count of Feedback')
plt.title('Feedback Count by Course Difficulty')
ax.bar_label(ax.containers[0], fmt='%.0f', label_type='edge')
plt.show();

# %%
# Histogram for Study Hours (per week)
plt.figure(figsize=(10, 6))
ax = sns.histplot(data=df, x='Study Hours (per week)', bins=20)
plt.xlabel('Study Hours (per week)')
plt.ylabel('Count of Students')
plt.title('Distribution of Study Hours')
plt.show()

# %%
# Histogram for Overall Satisfaction
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Overall Satisfaction')
plt.xlabel('Overall Satisfaction')
plt.ylabel('Count of Students')
plt.title('Distribution of Overall Satisfaction')
plt.show()

# %%
# Word Frequency Analysis
from collections import Counter
word_frequency = Counter(" ".join(df['Processed_Feedback']).split()).most_common(30)
plt.figure(figsize=(20, 10))
color = sns.color_palette()[0]
ax = sns.barplot(x=[word[1] for word in word_frequency], y=[word[0] for word in word_frequency], color=color)
ax.bar_label(ax.containers[0], fmt='%.0f', label_type='edge')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Top 30 Most Frequently Used Words')
plt.show()

# %%
# Sentiment Box Plots
plt.figure(figsize=(10, 6))
color = sns.color_palette()[0]
sns.boxplot(data=df, x='Course Code', y='Sentiment_Scores', color=color)
plt.xlabel('Course Code')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Distribution by Course Code')
plt.xticks(rotation=45)
plt.show()

# %%
# Distribution of length of preocessed feedback
plt.figure(figsize=(10, 6))
sns.histplot(df['Char_Count'], kde = True, palette = 'hls')
plt.title('Distribution of length of preocessed feedback')
plt.xlabel('Length')
plt.ylabel('Count')
plt.show()

# %%
# Bar plot for Course Difficulty
plt.figure(figsize=(10, 6))
hue_order = ['Easy', 'Moderate', 'Challenging', 'Difficult']
sns.countplot(data=df, x='Course Code', hue='Course Difficulty', hue_order=hue_order)
plt.xlabel('Course Difficulty')
plt.ylabel('Count of Feedback')
plt.title('Feedback Count by Course Difficulty')
plt.legend(loc=1)
plt.show();

# %%
# Bar plot for Course Code distribution by Sentiment distribution
plt.figure(figsize=(10, 6))
# hue_order = ['positive', 'neutral', 'negative']
hue_order = ['positive', 'negative']
sns.countplot(data=df, x='Course Code', hue='Sentiments', hue_order=hue_order)
plt.xlabel('Course Code')
plt.ylabel('Count of Feedback')
plt.title('Course Code distribution by Sentiment distribution')
plt.legend(loc=1)
plt.show();

# %%
# Sentiment Distribution by Course Difficulty
plt.figure(figsize=(10, 6))
# hue_order = ['positive', 'neutral', 'negative']
hue_order = ['positive', 'negative']
order = ['Easy', 'Moderate', 'Challenging', 'Difficult']
sns.countplot(data=df, x='Course Difficulty', hue='Sentiments', hue_order=hue_order, order=order)
plt.xlabel('Course Difficulty')
plt.ylabel('Count of Feedback')
plt.title('Sentiment Distribution by Course Difficulty')
plt.show()

# %%
# Sentiment Distribution by Gender
plt.figure(figsize=(10, 6))
# hue_order = ['positive', 'neutral', 'negative']
hue_order = ['positive', 'negative']
sns.countplot(data=df, x='Gender', hue='Sentiments', hue_order=hue_order)
plt.xlabel('Gender')
plt.ylabel('Count of Feedback')
plt.title('Sentiment Distribution by Gender')
plt.show()

# %%
# Word Count distribution by course difficulty
plt.figure(figsize=(10, 6))
order = ['Easy', 'Moderate', 'Challenging', 'Difficult']
sns.boxplot(data=df, x='Course Difficulty', y='Word_Count', order=order)
plt.xlabel('Course Difficulty')
plt.ylabel('Word Count')
plt.title('Distribution of Word Count for different levels of Course Difficulty')
plt.show()

# %%
# Distribution of Study Hours (per week) and Overall Satisfaction
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Study Hours (per week)', y='Overall Satisfaction', ci=None)
plt.xlabel('Study Hours (per week)')
plt.ylabel('Overall Satisfaction')
plt.title('Distribution of Study Hours (per week) and Overall Satisfaction')
plt.show()

# %%
# scatter plot of character count vs. word count
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df,x=df['Char_Count'],y= df['Word_Count'], alpha=0.5, color=color)
plt.xlabel('Character Count')
plt.ylabel('Word Count')
plt.title('Character Count vs. Word Count in Tweets')
plt.show()

# %%
# Sentiment vs. Overall Satisfaction
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sentiment_Scores', y='Overall Satisfaction', data=df)
plt.xlabel('Sentiment Score')
plt.ylabel('Overall Satisfaction')
plt.title('Sentiment score vs. Overall Satisfaction')
# plt.xticks(np.arange(-1, 1.1, 0.5))
# plt.yticks(np.arange(0, 11))
plt.show()

# %%
# Sentiment score Distribution by Course code
plt.figure(figsize=(10, 6))
sns.relplot(data=df, x='Course Code',y = 'Sentiment_Scores', kind='scatter')
plt.xlabel('Course code')
plt.ylabel('Sentiment Scores')
plt.title('Sentiment score vs Course code distribution')
plt.xticks(rotation=45)
plt.show();

# %%
# correlation matrix of numerical variables in the data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
# Select only numeric columns
numeric_df = df.select_dtypes(include=[float, int])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(20, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", vmin=-1, vmax=1, linewidths=0.5, cmap='coolwarm')
plt.title('Correlation between variables in the dataset')

colorbar = plt.gca().collections[0].colorbar
colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
colorbar.set_ticklabels(['Strong Negative', 'Negative', 'Neutral', 'Positive', 'Strong Positive'])

plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

# %%
sns.set(style='ticks')
sns.pairplot(data=df, vars=['Study Hours (per week)', 'Overall Satisfaction', 
                            'Sentiment_Scores'], hue='Previous Experience', markers='o')
plt.suptitle('Study Hours (per week), Overall Satisfaction and Sentiment Score Distributions by Previous Experience',
             y=1.08)
plt.show();

# %% [markdown]
# ## Machine Learning Model
# We build a ML model for predicting sentiment labels.

# %%
X = df['Processed_Feedback']
y = df['Sentiments']

print(len(X), ',', len(y))

# %%
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print('Encoded Target Labels:')
print(y_encoded, '\n')

# get mapping for each label
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print('Label Mappings:')
print(le_name_mapping)

# %%
# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %% [markdown]
# ### XGBoost Classifier

# %%
# Using random train and test subsets

# Preprocessor
preprocessor = Pipeline([
    ('bow', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
])

# Create a model
model = xgb.XGBClassifier(
    eta=0.01,
    max_depth=7,
    n_estimators=100,
    objective='binary:logistic',
    eval_metric='logloss',
)

# Create a pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model),
])

# Fit the pipeline to the train data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred_xgb = pipeline.predict(X_test)

# Calculate the accuracy score
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

# Calculate the recall score
recall_xgb = recall_score(y_test, y_pred_xgb)

# Calculate the precision score
precision_xgb = precision_score(y_test, y_pred_xgb)

# Calculate the f1-score
f1_xgb = f1_score(y_test, y_pred_xgb)

# Print the results
print('Accuracy score: {:.3f} %'.format(accuracy_xgb * 100))
print('Recall score: {:.3f}'.format(recall_xgb))
print('Precision score: {:.3f}'.format(precision_xgb))
print('F1-score: {:.3f}'.format(f1_xgb))

# %%
# using cross validation techniques

# Preprocessor
preprocessor = Pipeline([
    ('bow', CountVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
])

# Model
model_cv = xgb.XGBClassifier(
    eta=0.01,
    max_depth=10,
    n_estimators=100,
    objective='binary:logistic',
    eval_metric='logloss',
)


# Pipeline
pipeline_cv = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model_cv),
])

kf = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(pipeline_cv, X, y_encoded, cv=kf)
mean_cv_score = cv_scores.mean()
print('Cross-validation accuracy score: {:.3f} %'.format(mean_cv_score*100))

# %%
# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative',  'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Support Vector Machine

# %%
# Preprocessor
preprocessor = Pipeline([
    ('bow', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
])

# Model (SVM Classifier)
svm_model = SVC()

# Pipeline
pipeline_svm = Pipeline([
    ('preprocessor', preprocessor),
    ('model', svm_model),
])

pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)

# Calculate the accuracy score
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Calculate the recall score
recall_svm = recall_score(y_test, y_pred_svm)

# Calculate the precision score
precision_svm = precision_score(y_test, y_pred_svm)

# Calculate the f1-score
f1_svm = f1_score(y_test, y_pred_svm)

# Print the results
print('Accuracy score: {:.3f} %'.format(accuracy_svm * 100))
print('Recall score: {:.3f}'.format(recall_svm))
print('Precision score: {:.3f}'.format(precision_svm))
print('F1-score: {:.3f}'.format(f1_svm))

# %%
# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative',  'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Random Forest Classifier

# %%
from sklearn.ensemble import RandomForestClassifier

# Preprocessor
preprocessor = Pipeline([
    ('bow', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
])

random_forest = RandomForestClassifier()

# Pipeline
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', random_forest),
])

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

# Calculate the accuracy score
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Calculate the recall score
recall_rf = recall_score(y_test, y_pred_rf)

# Calculate the precision score
precision_rf = precision_score(y_test, y_pred_rf)

# Calculate the f1-score
f1_rf = f1_score(y_test, y_pred_rf)

# Print the results
print('Accuracy score: {:.3f} %'.format(accuracy_rf * 100))
print('Recall score: {:.3f}'.format(recall_rf))
print('Precision score: {:.3f}'.format(precision_rf))
print('F1-score: {:.3f}'.format(f1_rf))

# %%
# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative',  'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Naive Bayes Classifier

# %%
from sklearn.naive_bayes import MultinomialNB

# Preprocessor
preprocessor = Pipeline([
    ('bow', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
])

naive_bayes = MultinomialNB()

# Pipeline
pipeline_nb = Pipeline([
    ('preprocessor', preprocessor),
    ('model', naive_bayes),
])

pipeline_nb.fit(X_train, y_train)
y_pred_nb = pipeline_nb.predict(X_test)

# Calculate the accuracy score
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Calculate the recall score
recall_nb = recall_score(y_test, y_pred_nb)

# Calculate the precision score
precision_nb = precision_score(y_test, y_pred_nb)

# Calculate the f1-score
f1_nb = f1_score(y_test, y_pred_nb)

# Print the results
print('Accuracy score: {:.3f} %'.format(accuracy_nb * 100))
print('Recall score: {:.3f}'.format(recall_nb))
print('Precision score: {:.3f}'.format(precision_nb))
print('F1-score: {:.3f}'.format(f1_nb))

# %%
# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative',  'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### **Comparison of Accuracy Scores of all Models used.**

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Define the accuracies of the models (assuming these variables are already defined)
accuracy_xgb = 0.85  # Example value
accuracy_svm = 0.80  # Example value
accuracy_nb = 0.75  # Example value
accuracy_rf = 0.82  # Example value

# Plotting the bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x=['XGBoost', 'SVM', 'Naive Bayes Classifier', 'RandomForest Classifier'], 
            y=[accuracy_xgb*100, accuracy_svm*100, accuracy_nb*100, accuracy_rf*100],
            palette='Blues_d')  # Use a color palette for better visualization

plt.xlabel('Machine Learning Model')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Accuracy Scores of the Models')
plt.suptitle('Sentiment Analysis on Student Feedback in Computer Engineering, 300L', y=0.94)
plt.show()

# %% [markdown]
# #### **Make Predictions on Random Data.**

# %%
print('Label Mappings:')
print(le_name_mapping)

pipeline.fit(X, y_encoded)
sentiment_categories = {0: 'negative', 1: 'positive'}
print(sentiment_categories)

# %%
new_examples = [
    "I swear, I hate akanni with him crazy course",
    "Adebayo go just dy stress person life mtcheww",
    "Practicals are surreal in 381",
    "Why we still dy get 8'o clock class? lol"
    "Una don craze for this department",
    "raqibcodes sef na low-key scholar",
    "I love coding a lot",
    "good"
]

# %%
# XGBoost

predicted_sentiments = pipeline.predict(preprocess_text(example) for example in new_examples)
predicted_sentiment_labels = [sentiment_categories[sentiment] for sentiment in predicted_sentiments]

print("Predicted Sentiments:", predicted_sentiment_labels)

# %%
# SVM

predicted_sentiments = pipeline_svm.predict(preprocess_text(example) for example in new_examples)
predicted_sentiment_labels = [sentiment_categories[sentiment] for sentiment in predicted_sentiments]

print("Predicted Sentiments:", predicted_sentiment_labels)

# %%
# NB

predicted_sentiments = pipeline_nb.predict(preprocess_text(example) for example in new_examples)
predicted_sentiment_labels = [sentiment_categories[sentiment] for sentiment in predicted_sentiments]

print("Predicted Sentiments:", predicted_sentiment_labels)

# %%
# export data to csv to perform further text classification with BERT and KerasNLP
df.to_csv('exported_sentiments.csv', index=False)

# %%

