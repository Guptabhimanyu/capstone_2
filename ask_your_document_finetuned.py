import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import finetuned_flan_t5_large

# Define a function to generate a response to a question using the finetuned model


def generate_response(question, model):
    inputs = model.prepare_inputs_for_generation(
        tokenizer(question, return_tensors="pt")["input_ids"], max_length=10000)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Load the finetuned model
model = finetuned_flan_t5_large.AutoModelForQuestionAnswering.from_pretrained(
    "finetuned_flan-t5-large")
tokenizer = finetuned_flan_t5_large.AutoTokenizer.from_pretrained(
    "google/flan-t5-large")

# Display the header
st.header("Ask Your Document")

# Display the sidebar
uploaded_file = st.sidebar.file_uploader(
    "Upload your document (PDF, Word, Excel, or Presentation)", type=["pdf", "docx", "xlsx", "pptx"])
user_question = st.sidebar.text_input(
    "Ask a question about your document:")

# Process the uploaded file and generate a response to the user's question
if uploaded_file is not None:
    # ...
    response = generate_response(user_question, model)

    # Display the response
    st.write(response)
