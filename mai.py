from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

app = FastAPI()

df = pd.read_csv("mentahealth.csv")

model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

@app.post("/answer/")
def answer_question(context: str):
    input_ids = tokenizer.encode(context, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids,num_return_sequences=1,max_length=70, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        first_position = generated_text.find('\n')
        second_position = generated_text.find('\n', first_position + 1)
        gen_text=generated_text[second_position+1:]
        generated_text=gen_text.replace("...","")
        mtext=generated_text.replace("\n","")
        last_period_position = mtext.rfind('.')
        print (generated_text)
    return mtext[:last_period_position+1]
