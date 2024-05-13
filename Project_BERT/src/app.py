import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Define the sentiment labels
sentiment_labels = {0: "positive", 1: "neutral", 2: "negative"}
message_types = {"Negative": "error", "Neutral": "info", "Positive": "success"}

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#Image input:
def rounded_image_with_name(image_path, name):
    st.markdown(
        f'<style>img.rounded{{border-radius: 10px;}}</style>',
        unsafe_allow_html=True
    )
    st.sidebar.image(image_path, width=80, caption=name, output_format='PNG')


# Create a function to check if input is valid
def is_valid_input(text):
    return bool(text.strip())  # Check if the text is not empty after stripping whitespace

# Create a function to predict sentiment
def predict_sentiment(text, model, tokenizer):
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )

    input_ids = encoded_review["input_ids"]
    attention_mask = encoded_review["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(output.logits, dim=1).numpy().squeeze()
        prediction = np.argmax(probabilities)

    return sentiment_labels[prediction], probabilities

# Create the Streamlit user interface
st.set_page_config(page_title="Sentiment Analysis with BERT", page_icon=":sunny:")

st.title("Sentiment Analysis with BERT")
st.markdown(
    """
    This application analyzes the sentiment of text using BERT (Bidirectional Encoder Representations from Transformers).
    """
)

st.sidebar.title("RCC Institute of Information and Technology")
st.sidebar.markdown("Project Information")
st.sidebar.info(
    """
    This project is created by Group-18 under the mentorship of Assistant professor Dr. Dipankar Majumder as part of the Sentiment Analysis project.
    """
)

st.sidebar.write("Project Members:")
# Example usage
rounded_image_with_name("/content/drive/MyDrive/Project_BERT/images/rittick.png", "Rittick Roy CSE2021L14")
rounded_image_with_name("/content/drive/MyDrive/Project_BERT/images/pratanu.png", "Pratanu Ghorui CSE2021L13")
rounded_image_with_name("/content/drive/MyDrive/Project_BERT/images/arghya.png", "Arghya Chowdhury CSE2021L02")
rounded_image_with_name("/content/drive/MyDrive/Project_BERT/images/rittick.png", "Anuradha Adhikari CSE2021L16")

input_text = st.text_area("Enter your text here:", height=200)

if st.button("Analyze"):
    if not is_valid_input(input_text):
        st.error("Please enter valid text.")
    else:
        prediction, probabilities = predict_sentiment(input_text, model, tokenizer)
        formatted_probabilities = [f"{prob * 100:.2f}%" for prob in probabilities]

        st.subheader("Result")
        st.write("Sentiment:", prediction)
        st.write(
            "Prediction Probabilities:",
            {label: prob for label, prob in zip(sentiment_labels.values(), formatted_probabilities)},
        )
