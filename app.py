import streamlit as st
from transformers import pipeline, set_seed

st.title("Next Sentence Prediction using Generative AI")
st.write("Enter a sentence and let GPT-2 predict the most likely next sentence.")

input_text = st.text_input("Enter your sentence:")

generator = pipeline('text-generation', model='gpt2')
set_seed(42)

if input_text:
    st.subheader("Predicted Sentences:")
    results = generator(input_text, max_length=50, num_return_sequences=3)
    for i, result in enumerate(results):
        st.write(f"{i+1}. {result['generated_text']}")
