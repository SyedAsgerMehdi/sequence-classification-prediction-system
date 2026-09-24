import os, json, random, requests
import numpy as np
import tensorflow as tf

DATA_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def load_text(path="data/input.txt"):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path)>1000:
        return open(path,encoding="utf-8").read()
    try:
        r=requests.get(DATA_URL,timeout=30); r.raise_for_status()
        text=r.text
        open(path,"w",encoding="utf-8").write(text)
        return text
    except Exception:
        return open("data/sample.txt",encoding="utf-8").read()

def prepare_sequences(text, seq_length=60, step=3, max_sequences=20000):
    chars=sorted(set(text))
    c2i={c:i for i,c in enumerate(chars)}
    i2c=np.array(chars)
    encoded=np.array([c2i[c] for c in text],dtype=np.int32)
    starts=list(range(0,max(1,len(encoded)-seq_length),step))[:max_sequences]
    X=np.array([encoded[s:s+seq_length] for s in starts],dtype=np.int32)
    y=np.array([encoded[s+seq_length] for s in starts],dtype=np.int32)
    return X,y,c2i,i2c

def split_data(X,y):
    split=int(len(X)*.9)
    return X[:split],X[split:],y[:split],y[split:]

def save_json(obj,path):
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2)

def sample(probs,temp=.8):
    p=np.asarray(probs,dtype="float64")
    p=np.log(np.maximum(p,1e-9))/max(temp,.01)
    p=np.exp(p-np.max(p)); p/=p.sum()
    return np.random.choice(len(p),p=p)

def generate(model,seed,c2i,i2c,length=300,temp=.8):
    ids=[c2i[c] for c in seed if c in c2i]
    if not ids: ids=[0]
    out=seed; n=model.input_shape[1]
    for _ in range(length):
        ctx=ids[-n:]
        if len(ctx)<n: ctx=[ids[0]]*(n-len(ctx))+ctx
        p=model.predict(np.array([ctx]),verbose=0)[0]
        nxt=int(sample(p,temp)); ids.append(nxt); out+=str(i2c[nxt])
    return out
