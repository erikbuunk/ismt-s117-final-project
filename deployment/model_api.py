# run:  uvicorn model_api:app
# /docs for endpoints: http://127.0.0.1:8000/docs
# /frontend/index.html for the website


from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sklearn.svm import SVC

import pandas as pd
import pickle
import uuid
import torch
import re

import transformers

from transformers import BertTokenizer, BertModel

print("loading pretrained BERT feature model")
MODEL_NAME = 'bert-base-uncased'

# Load pre-trained model
model = BertModel.from_pretrained(MODEL_NAME)
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

path = "./data/"
device = torch.device("cpu")
filename = path + "user_lyrics_data.csv"

print("Loading Trained Prediction Model")
prediction_model = pd.read_pickle(path + 'final_lyrics_model.pkl.gz')


def clean_single_lyric(text):
    x = text.strip()
    y = re.sub('\[.*?\]', '', x)
    z = re.sub('\(.*?\)', '', y)
    z1 = pd.DataFrame([z], columns=["Lyric"])
    return z1


def generate_single_BERT_token(text):
    tokens = tokenizer.batch_encode_plus(
        text["Lyric"],
        pad_to_max_length=True,
        return_tensors="pt",
        max_length=512,
        truncation=True)
    tokens.to(device)
    outputs = model(**tokens)
    o = outputs[0][:, 0].cpu().detach().numpy()
    return o


def get_prediction(token):
    p = prediction_model.predict(token)
    print("Prediction: " + p)
    return p


def generate_id():
    uuidOne = uuid.uuid1()
    return uuidOne.hex


def write_prediction(cleaned_lyrics, token, prediction, id):
    df = pd.DataFrame.from_dict(
        {"id": [id], "Prediction": prediction, "Truth": '', "Lyric": cleaned_lyrics})
    df_token = pd.DataFrame(token)
    df_record = pd.concat([df, df_token], axis=1)

    try:
        with open(filename) as f:
            df_in = pd.read_csv(filename)
            if len(df_in[df_in["id"] == id]) == 0:
                df_in = df_in.append(df_record)
                df_in.to_csv(filename, index=False)
    except IOError:
        df_record.to_csv(filename, index=False)

    return


def update_prediction(id, new_value):
    df_in = pd.read_csv(filename)
    if len(df_in[df_in["id"] == id]) == 1:
        df_in.loc[df_in["id"] == id, ["Truth"]] = new_value
        df_in.to_csv(filename, index=False)


def predict(lyric):
    text = clean_single_lyric(lyric)
    token = generate_single_BERT_token(text)
    predicted_genre = get_prediction(token)
    id = generate_id()
    raw_text = list(text["Lyric"])
    write_prediction(raw_text, token, predicted_genre, id)
    return (id, predicted_genre)


app = FastAPI()


app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/predict/{lyric}")
async def root(lyric: str):
    (idx, prediction) = predict(lyric)
    return {"message": str(prediction[0]), "id": str(idx)}


@app.get("/update/{idx}/{truth}")
async def root(idx: str, truth: str):
    update_prediction(idx, truth)
    return {"message": "Success"}
