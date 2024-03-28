import streamlit as st
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import time

page_element = """<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://wallup.net/wp-content/uploads/2019/09/859126-poster-movie-film-movies-posters.jpg");
  background-size:cover;
  
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)

# Load the IMDb dataset
imdb_df = pd.read_csv('imdb_movie_reviews.csv')

# Load the trained model
model = tf.keras.models.load_model('movie_review_sentiment_analysis2.keras')

# Load the Tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(imdb_df['review'])

# Define mapping for sentiment labels
sentiment_map = {1: "Positive review", 0: "Negative review"}

# Function to preprocess text
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=256)
    return padded_sequence

# Function to predict sentiment label
def predict_sentiment_label(text):
    start_time = time.time()
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)[0][0]
    sentiment_label = sentiment_map[int(round(prediction))]
    end_time = time.time()
    runtime = end_time - start_time
    
    # Convert sentiment label to numerical value
    sentiment_value = 1 if sentiment_label == 'Positive review' else 0
    
    # Calculate loss
    true_label = imdb_df['sentiment'][0]
    true_value = 1 if true_label == 'positive' else 0
    loss = abs(prediction - true_value)
    
#      # Calculate loss
#     loss = abs(prediction - imdb_df['sentiment'][0])
    
    return sentiment_label, runtime, loss

#     processed_text = preprocess_text(text)
#     prediction = model.predict(processed_text)[0][0]
#     sentiment_label = sentiment_map[int(round(prediction))]
#     return sentiment_label

def calculate_accuracy():
    correct_predictions = 0
    total_predictions = len(imdb_df)
    
    for index, row in imdb_df.iterrows():
        text = row['review']
        true_label = row['sentiment']
        predicted_label, _, _ = predict_sentiment_label(text)
        if true_label == predicted_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy


# Define the Streamlit app
def main():
    st.title("IMDb Movie Review Sentiment Analysis")
    review_text = st.text_area("Enter your movie review here:")
    # selected_review = st.selectbox("Select a Review", imdb_df['review'])

    # Button to predict sentiment
    if st.button("Predict"):
        if review_text.strip() == "":
            st.error("Please select a movie review.")
        else:
            # Predict sentiment label
            sentiment_label,runtime,loss = predict_sentiment_label(review_text)
            st.success(f"Sentiment: **{sentiment_label}**")
            st.success(f"Runtime: {runtime:.4f} seconds")
            st.error(f"Loss: {loss:.4f}")
            
            # Calculate and display accuracy
            accuracy = calculate_accuracy()
            st.success(f"Accuracy: {accuracy:.4f}")
            
#             if sentiment_label == "Positive review":
#                 st.success("Sentiment: **Positive review**")
#             else:
#                 st.error("Sentiment: **Negative review**")
    

if __name__ == "__main__":
    main()
