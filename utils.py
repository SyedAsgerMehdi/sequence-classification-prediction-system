import os, json, random, re, requests, unicodedata
import numpy as np
import tensorflow as tf

DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def load_text(path="data/input.txt"):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path)>1000:
        return clean_text(open(path,encoding="utf-8").read())
    try:
        r=requests.get(DATA_URL,timeout=30); r.raise_for_status()
        text=r.text
        open(path,"w",encoding="utf-8").write(text)
        return clean_text(text)
    except Exception:
        return clean_text(open("data/sample.txt",encoding="utf-8").read())

def clean_text(text):
    """Normalize source text while preserving meaningful line breaks."""
    text=unicodedata.normalize("NFKC",text).replace("\r\n","\n").replace("\r","\n")
    text=text.replace("\t"," ").replace("\x00","")
    text="".join(character for character in text if character in "\n" or 32<=ord(character)<=126)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

def prepare_sequences(text, seq_length=60, step=3, max_sequences=20000):
    text=clean_text(text)
    chars=sorted(set(text))
    c2i={c:i for i,c in enumerate(chars)}
    i2c=np.array(chars)
    encoded=np.array([c2i[c] for c in text],dtype=np.int32)
    if len(encoded)<=seq_length:
        raise ValueError("The cleaned dataset must be longer than seq_length.")
    starts=list(range(0,len(encoded)-seq_length,step))[:max_sequences]
    X=np.array([encoded[s:s+seq_length] for s in starts],dtype=np.int32)
    y=np.array([encoded[s+seq_length] for s in starts],dtype=np.int32)
    return X,y,c2i,i2c

def split_data(X,y):
    train_end=int(len(X)*.8)
    validation_end=int(len(X)*.9)
    return (
        X[:train_end],X[train_end:validation_end],X[validation_end:],
        y[:train_end],y[train_end:validation_end],y[validation_end:]
    )

def save_json(obj,path):
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2)

def predict_next_character(model,seed,c2i,i2c,top_k=5):
    ids=[c2i[c] for c in seed if c in c2i]
    if not ids:
        raise ValueError("The prompt must contain at least one known character.")
    n=model.input_shape[1]
    ctx=ids[-n:]
    if len(ctx)<n:
        ctx=[ids[0]]*(n-len(ctx))+ctx
    probabilities=model.predict(np.array([ctx]),verbose=0)[0]
    indices=np.argsort(probabilities)[::-1][:top_k]
    return [(str(i2c[index]),float(probabilities[index])) for index in indices]

def display_character(character):
    """Use readable labels for whitespace characters in the UI and CLI."""
    return {" ":"[space]","\n":"[newline]"}.get(character,character)
