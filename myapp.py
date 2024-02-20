# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import streamlit as st

page_element="""<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://wallup.net/wp-content/uploads/2019/09/859126-poster-movie-film-movies-posters.jpg");
  background-size:cover;
  
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)[0][0]
    sentiment_label = sentiment_map[int(round(prediction))]
    return sentiment_label

# Define the Streamlit app
def main():
    st.title("IMDb Movie Review Sentiment Analysis")
    review_text = st.text_area("Enter your movie review here:")
    # selected_review = st.selectbox("Select a Review", imdb_df['review'])

    # Button to predict sentiment
    if st.button("Predict"):
        if selected_review.strip() == "":
            st.error("Please select a movie review.")
        else:
            # Predict sentiment label
            sentiment_label = predict_sentiment_label(selected_review)
            if sentiment_label == "Positive review":
                st.success("Sentiment: **Positive review**")
            else:
                st.error("Sentiment: **Negative review**")

if __name__ == "__main__":
    main()

# import streamlit as st
# from streamlit.logger import get_logger

# LOGGER = get_logger(__name__)


# def run():
#     st.set_page_config(
#         page_title="Hello",
#         page_icon="👋",
#     )

#     st.write("# Welcome to Streamlit! 👋")

#     st.sidebar.success("Select a demo above.")

#     st.markdown(
#         """
#         Streamlit is an open-source app framework built specifically for
#         Machine Learning and Data Science projects.
#         **👈 Select a demo from the sidebar** to see some examples
#         of what Streamlit can do!
#         ### Want to learn more?
#         - Check out [streamlit.io](https://streamlit.io)
#         - Jump into our [documentation](https://docs.streamlit.io)
#         - Ask a question in our [community
#           forums](https://discuss.streamlit.io)
#         ### See more complex demos
#         - Use a neural net to [analyze the Udacity Self-driving Car Image
#           Dataset](https://github.com/streamlit/demo-self-driving)
#         - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
#     """
#     )


# if __name__ == "__main__":
#     run()
