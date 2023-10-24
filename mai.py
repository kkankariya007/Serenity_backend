from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

app = FastAPI()

# Load the mental health dataset
df = pd.read_csv("mentahealth.csv")

model_name = "gpt2"  # You can replace this with the model of your choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

@app.post("/answer/")
def answer_question(context: str):
    # Tokenize the context and convert it to tensor
    input_ids = tokenizer.encode(context, return_tensors="pt")

    # Generate an answer using the model
    with torch.no_grad():
        output = model.generate(input_ids,num_return_sequences=1,max_length=40, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"answer": generated_text}
