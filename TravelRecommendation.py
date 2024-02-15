#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from textblob import TextBlob


# In[2]:


# Load Data
travel_path = "C:/Users/josue/Documents/UNIVERSITY/FYP/df_travel.xlsx"
restaurants_path = "C:/Users/josue/Documents/UNIVERSITY/FYP/df_restaurants.xlsx"
booking_path = "C:/Users/josue/Documents/UNIVERSITY/FYP/df_booking.xlsx"


# In[3]:


travel_df = pd.read_excel(travel_path)
restaurants_df = pd.read_excel(restaurants_path, nrows = 100)
booking_df = pd.read_excel(booking_path, nrows = 100)


# In[4]:


print("Travel Dataset:")
print(travel_df.head())

print("\nRestaurants Dataset:")
print(restaurants_df.head())

print("\nBooking Satisfaction Dataset:")
print(booking_df.head())


# In[5]:


print("Travel Dataset Statistics:")
print(travel_df.describe())

print("\nRestaurants Dataset Statistics:")
print(restaurants_df.describe())

print("\nBooking Satisfaction Dataset Statistics:")
print(booking_df.describe())


# In[6]:


# Print missing values of the datasets
print("Travel Dataset Missing Values:")
print(travel_df.isnull().sum())

print("\nRestaurants Dataset Missing Values:")
print(restaurants_df.isnull().sum())

print("\nBooking Satisfaction Dataset Missing Values:")
print(booking_df.isnull().sum())


# In[7]:


# Cleaning travel information dataset
travel_df.dropna(inplace=True)

# Convert 'Start date' to datetime format
travel_df['Start date'] = pd.to_datetime(travel_df['Start date'])

# Extract year and month from 'Start date'
travel_df['YearMonth'] = travel_df['Start date'].dt.to_period('M')

# Clean and convert 'Accommodation cost' and 'Transportation cost' to numeric
travel_df['Accommodation cost'] = travel_df['Accommodation cost'].apply(lambda x: pd.to_numeric(''.join(filter(str.isdigit, str(x))), errors='coerce') if pd.notna(x) else x)
travel_df['Transportation cost'] = travel_df['Transportation cost'].apply(lambda x: pd.to_numeric(''.join(filter(str.isdigit, str(x))), errors='coerce') if pd.notna(x) else x)

clean_travel_df = travel_df.copy()

# Display the cleaned DataFrame
print("Cleaned Travel DataFrame:")
print(clean_travel_df)


# In[8]:


# Cleaning restaurants dataset
restaurants_df.drop(['latitude', 'longitude'], axis=1, inplace=True)

# Replace "â‚¬" with "£" in 'price_level' and 'price_range' column
restaurants_df['price_level'] = restaurants_df['price_level'].str.replace('â‚¬', '£')
restaurants_df['price_range'] = restaurants_df['price_range'].str.replace('â‚¬', '£')

# Drop rows where both 'price_level' and 'price_range' are missing
clean_restaurants_df = restaurants_df.dropna(subset=['price_level', 'price_range'], how='all')

# Display the cleaned DataFrame
print("Cleaned Restaurants DataFrame:")
print(clean_restaurants_df)


# In[9]:


# Travel Information Exploratory
# Histogram of Traveler Ages
plt.figure(figsize=(8, 6))
sns.histplot(clean_travel_df['Traveler age'], bins=20, kde=True)
plt.title('Distribution of Traveler Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of destinations 
plt.figure(figsize=(12, 6))
destination_counts = clean_travel_df['Destination'].value_counts()
sns.barplot(x=destination_counts.index, y=destination_counts.values, palette='viridis')
plt.title('Distribution of Destinations')
plt.xlabel('Destination')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, ha='right')
plt.show()

# Plot the distribution of travel durations
plt.figure(figsize=(10, 6))
sns.histplot(clean_travel_df['Duration (days)'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Travel Durations')
plt.xlabel('Duration (days)')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of traveler ages
plt.figure(figsize=(10, 6))
sns.histplot(clean_travel_df['Traveler age'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Traveler Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot the trends in booking frequency over time
plt.figure(figsize=(12, 6))
booking_trends = clean_travel_df['YearMonth'].value_counts().sort_index()
sns.lineplot(x=booking_trends.index.astype(str), y=booking_trends.values, marker='o', color='skyblue')
plt.title('Booking Frequency Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45, ha='right')
plt.show()

# Extract month from 'Start date'
clean_travel_df['Month'] = clean_travel_df['Start date'].dt.month_name()

# Plot the distribution of bookings across months
plt.figure(figsize=(12, 6))
booking_monthly_distribution = clean_travel_df['Month'].value_counts().sort_index()
sns.barplot(x=booking_monthly_distribution.index, y=booking_monthly_distribution.values, palette='viridis')
plt.title('Distribution of Bookings Across Months')
plt.xlabel('Month')
plt.ylabel('Number of Bookings')
plt.show()

# Plot a histogram of travel costs
plt.figure(figsize=(12, 6))
sns.histplot(clean_travel_df['Accommodation cost'] + travel_df['Transportation cost'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Travel Costs')
plt.xlabel('Total Travel Cost')
plt.ylabel('Frequency')
plt.show()


# In[10]:


# Restaurants Exploratory
# Identify top cuisines
top_cuisines = restaurants_df['cuisines'].value_counts().head(10)

# Plot the distribution of top cuisines
plt.figure(figsize=(12, 6))
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis')
plt.title('Top Cuisines')
plt.xlabel('Number of Restaurants')
plt.ylabel('Cuisine')
plt.show()

# Plot the distribution of restaurants based on price levels
plt.figure(figsize=(8, 6))
sns.countplot(x='price_level', data=restaurants_df, palette='viridis')
plt.title('Distribution of Restaurants Based on Price Levels')
plt.xlabel('Price Level')
plt.ylabel('Number of Restaurants')
plt.show()

# Analyze the relationship between price levels and ratings
plt.figure(figsize=(12, 6))
sns.boxplot(x='price_level', y='avg_rating', data=restaurants_df, palette='viridis')
plt.title('Relationship Between Price Levels and Ratings')
plt.xlabel('Price Level')
plt.ylabel('Average Rating')
plt.show()

# Check the distribution of claimed awards
plt.figure(figsize=(8, 6))
sns.countplot(x='claimed', data=restaurants_df, palette='viridis')
plt.title('Distribution of Claimed Awards in Restaurants')
plt.xlabel('Claimed Award')
plt.ylabel('Number of Restaurants')
plt.show()

# Explore the impact of claimed awards on ratings
plt.figure(figsize=(12, 6))
sns.boxplot(x='claimed', y='avg_rating', data=restaurants_df, palette='viridis')
plt.title('Impact of Claimed Awards on Ratings')
plt.xlabel('Claimed Award')
plt.ylabel('Average Rating')
plt.show()


# In[11]:


# Reviews Exploratory
# Plot a histogram of customer satisfaction ratings
plt.figure(figsize=(10, 6))
sns.histplot(booking_df['satisfaction'], bins=5, kde=True, color='skyblue')
plt.title('Distribution of Customer Satisfaction Ratings')
plt.xlabel('Satisfaction Rating')
plt.ylabel('Frequency')
plt.show()


# In[12]:


# Store our clean datasets
clean_travel_df.to_excel("clean_travel_data.xlsx", index=False)
clean_restaurants_df.to_excel("clean_restaurants_data.xlsx", index=False)
booking_df.to_excel("clean_booking_data.xlsx", index=False)


# In[13]:


# Create a feature combining relevant columns for content-based recommendation
clean_travel_df['Features'] = clean_travel_df[['Traveler age', 'Traveler gender', 'Accommodation type', 'Transportation type']].astype(str).agg(' '.join, axis=1)

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_travel_df['Features'])

# Compute the cosine similarity matrix
cosine_similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Convert the similarity matrix into a DataFrame
cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=clean_travel_df['Trip ID'], columns=clean_travel_df['Trip ID'])

# Function to get similar items (destinations)
def get_similar_destinations(trip_id, threshold=0.5):
    similar_destinations = cosine_similarity_df[trip_id]
    similar_destinations = similar_destinations[similar_destinations > threshold].index
    return similar_destinations

# Function to recommend destinations 
def recommend_destinations(user_input, threshold=0.5):
    # Load the trained TF-IDF vectorizer
    tfidf_vectorizer = joblib.load('travel_tfidf_vectorizer.joblib')
    tfidf_matrix = joblib.load('travel_tfidf_matrix.joblib')
    cosine_similarity_df = joblib.load('travel_cosine_similarity_df.joblib')

    # Create a user input vector
    user_input_vector = tfidf_vectorizer.transform([user_input])

    # Calculate cosine similarity with existing destinations
    cosine_similarities = linear_kernel(user_input_vector, tfidf_matrix).flatten()

    # Get indices of similar destinations
    similar_destinations_indices = cosine_similarities.argsort()[:-6:-1]

    # Get the recommended destinations
    recommended_destinations = clean_travel_df.iloc[similar_destinations_indices]['Destination'].tolist()

    if not recommended_destinations:
        return ["No recommendations based on the given preferences."]
    
    return recommended_destinations

# Example
user_input = '30 female hotel car'
recommended_destinations = recommend_destinations(user_input)

print(recommended_destinations)

# Save the models
joblib.dump(tfidf_vectorizer, 'travel_tfidf_vectorizer.joblib')
joblib.dump(cosine_similarity_df, 'travel_cosine_similarity_df.joblib')
joblib.dump(tfidf_matrix, 'travel_tfidf_matrix.joblib')  # Add this line to save the TF-IDF matrix


# In[14]:


# Drop irrelevant columns
relevant_columns = ['restaurant_name', 'country', 'region', 'city', 'price_range', 'cuisines', 'avg_rating']
restaurant_data = clean_restaurants_df[relevant_columns]

# Handle missing values if any
restaurant_data = restaurant_data.dropna()

# Combine relevant features for TF-IDF processing
restaurant_data['combined_features'] = restaurant_data['country'] + ' ' + restaurant_data['region'] + ' ' + \
                                       restaurant_data['city'] + ' ' + restaurant_data['price_range'] + ' ' + \
                                       restaurant_data['cuisines'] + ' ' + restaurant_data['avg_rating'].astype(str)

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(restaurant_data['combined_features'])

# Compute the cosine similarity matrix
cosine_similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Convert the similarity matrix into a DataFrame
cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=restaurant_data['restaurant_name'], columns=restaurant_data['restaurant_name'])

# Function to get similar restaurants
def get_similar_restaurants(restaurant_name, threshold=0.5):
    similar_restaurants = cosine_similarity_df[restaurant_name]
    similar_restaurants = similar_restaurants[similar_restaurants > threshold].index
    return similar_restaurants

# Function to recommend restaurants based on user preferences
def recommend_restaurants(user_preferences, threshold=0.5):
    # Load the trained TF-IDF vectorizer
    tfidf_vectorizer = joblib.load('restaurant_tfidf_vectorizer.joblib')
    tfidf_matrix = joblib.load('restaurant_tfidf_matrix.joblib')
    cosine_similarity_df = joblib.load('restaurant_cosine_similarity_df.joblib')

    # Combine user preferences into a single text input
    user_input = ' '.join(user_preferences)

    # Transform the user input into a TF-IDF vector
    user_input_vector = tfidf_vectorizer.transform([user_input])

    # Calculate the cosine similarity between the user input and restaurants in the dataset
    cosine_similarities = linear_kernel(user_input_vector, tfidf_matrix).flatten()

    # Get indices of similar restaurants
    similar_restaurants_indices = cosine_similarities.argsort()[:-6:-1]

    # Get the recommended restaurants
    recommended_restaurants = restaurant_data.iloc[similar_restaurants_indices]['restaurant_name'].tolist()

    if not recommended_restaurants:
        return ["No recommendations based on the given preferences."]
    
    return recommended_restaurants

# Example
user_preferences = ['Italian', 'city_center', '$$$', 'high_rating']
recommendations = recommend_restaurants(user_preferences)

print(recommendations)

# Save the models
joblib.dump(tfidf_vectorizer, 'restaurant_tfidf_vectorizer.joblib')
joblib.dump(cosine_similarity_df, 'restaurant_cosine_similarity_df.joblib')
joblib.dump(tfidf_matrix, 'restaurant_tfidf_matrix.joblib')  # Add this line to save the TF-IDF matrix


# In[15]:


booking_data = booking_df.copy()

# Function to preprocess text
def preprocess_text(text):
    # Implement your preprocessing steps here
    # Example: Convert to lowercase
    return text.lower()
    
# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Classify the polarity as positive, negative, or neutral
    return 'positive' if analysis.sentiment.polarity > 0 else 'negative' if analysis.sentiment.polarity < 0 else 'neutral'

# Apply preprocessing
booking_data['processed_satisfaction'] = booking_data['satisfaction'].apply(preprocess_text)

# Apply sentiment analysis
booking_data['sentiment'] = booking_data['processed_satisfaction'].apply(analyze_sentiment)

print(booking_data[['satisfaction', 'processed_satisfaction', 'sentiment']])

# Save the processed data
booking_data.to_excel('processed_booking_data.xlsx', index=False)

