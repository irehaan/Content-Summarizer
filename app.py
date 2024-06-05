import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Load the models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
summarization_model_name = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)

# Define the summarization function
def summarize_document(document):
    inputs = tokenizer(document, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit app
st.title("Content Summarizer")
st.write("Enter a document below to get a summary.")

document = st.text_area("Document")

if st.button("Summarize"):
    if document:
        with st.spinner('Summarizing...'):
            summary = summarize_document(document)
            st.write("**Summary:**")
            st.write(summary)
    else:
        st.write("Please enter a document to summarize.")
