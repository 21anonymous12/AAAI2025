import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer 
import re
import os
import pandas as pd
import json
import requests


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_T5_base = "1anonymous1/MathSpeech_Ablation_Study_1stage_fine-tuned_with_errors_T5_base"

tokenizer_T5_base = T5Tokenizer.from_pretrained(path_T5_base)

model_T5_base = T5ForConditionalGeneration.from_pretrained(path_T5_base)

model_T5_base.to(device)
model_T5_base.eval()


def inference(text, model, tokenizer):
    input_text = f"translate text to LaTeX: {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=540,
        padding='max_length',
        truncation=True
    ).to('cuda')

    # Get correct sentence ids.
    corrected_ids = model.generate(
        inputs,
        max_length=275,
        num_beams=5, # `num_beams=1` indicated temperature sampling.
        early_stopping=True
    )

    # Decode.
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=False
    )
    return corrected_sentence

def postprocessing_T5(GT_raw):
    start_index = GT_raw.find("<pad>") + len("<pad>")
    end_index = GT_raw.find("</s>")
    GT_result = GT_raw[start_index:end_index]
    return GT_result


beam_result1 = "x plus 5y plus 10z école 0" # The 1st candidate for ASR, which contains an error
beam_result2 = "x plus 5y plus f z equals 0" # The 2nd candidate for ASR, which contains an error

result_list = []

input = f"{beam_result1} || {beam_result2}"
raw_T5_base = inference(input, model_T5_base, tokenizer_T5_base)
result_T5_base = postprocessing_T5(raw_T5_base)
result_T5_base = result_T5_base.replace(" ", "")
result_list.append(result_T5_base)
print(f"result : {result_T5_base}")
