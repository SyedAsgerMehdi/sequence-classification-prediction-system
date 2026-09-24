import os,time,argparse
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers,Model,callbacks
from utils import *

os.makedirs("outputs/models",exist_ok=True)

def build(kind,vocab,seq,emb=64,units=128):
    inp=layers.Input((seq,),dtype="int32")
    x=layers.Embedding(vocab,emb)(inp)
    if kind=="rnn": x=layers.SimpleRNN(units)(x)
    elif kind=="lstm": x=layers.LSTM(units)(x)
    else: x=layers.GRU(units)(x)
    x=layers.Dropout(.2)(x)
    out=layers.Dense(vocab,activation="softmax")(x)
    m=Model(inp,out)
    m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["sparse_categorical_accuracy"])
    return m

def train(kind,Xtr,ytr,Xv,yv,vocab,seq,epochs,batch):
    m=build(kind,vocab,seq)
    es=callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)
    start=time.time()
    h=m.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=epochs,batch_size=batch,callbacks=[es],verbose=1)
    elapsed=time.time()-start
    loss,acc=m.evaluate(Xv,yv,verbose=0)
    m.save(f"outputs/models/{kind}.keras")
    pd.DataFrame(h.history).to_csv(f"outputs/{kind}_history.csv",index=False)
    plt.figure(figsize=(7,4)); plt.plot(h.history["loss"],label="Train"); plt.plot(h.history["val_loss"],label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(kind.upper()+" Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(f"outputs/loss_{kind}.png",dpi=150); plt.close()
    return {"Model":kind.upper(),"Validation Accuracy":float(acc),"Validation Loss":float(loss),"Training Time (s)":round(elapsed,2),"Parameters":m.count_params()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=["rnn","lstm","gru","all"],default="all")
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--seq_length",type=int,default=60)
    ap.add_argument("--max_sequences",type=int,default=20000)
    a=ap.parse_args()
    set_seed(42)
    text=load_text()
    X,y,c2i,i2c=prepare_sequences(text,a.seq_length,3,a.max_sequences)
    Xtr,Xv,ytr,yv=split_data(X,y)
    kinds=["rnn","lstm","gru"] if a.model=="all" else [a.model]
    results=[]
    for k in kinds: results.append(train(k,Xtr,ytr,Xv,yv,len(c2i),a.seq_length,a.epochs,128))
    df=pd.DataFrame(results); df.to_csv("outputs/metrics.csv",index=False)
    save_json(c2i,"outputs/char_to_idx.json"); save_json({"characters":i2c.tolist()},"outputs/idx_to_char.json")
    plt.figure(figsize=(7,4)); plt.bar(df["Model"],df["Validation Accuracy"])
    plt.ylabel("Validation Accuracy"); plt.title("RNN Architecture Comparison"); plt.tight_layout()
    plt.savefig("outputs/comparison.png",dpi=150); plt.close()
    print(df.to_string(index=False))
if __name__=="__main__": main()
