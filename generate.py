import argparse,json,numpy as np
from tensorflow.keras.models import load_model
from utils import generate,set_seed

ap=argparse.ArgumentParser()
ap.add_argument("--model",choices=["rnn","lstm","gru"],default="lstm")
ap.add_argument("--seed",default="ROMEO: ")
ap.add_argument("--length",type=int,default=300)
ap.add_argument("--temperature",type=float,default=.8)
a=ap.parse_args()
set_seed(42)
m=load_model(f"outputs/models/{a.model}.keras")
c2i=json.load(open("outputs/char_to_idx.json",encoding="utf-8"))
i2c=np.array(json.load(open("outputs/idx_to_char.json",encoding="utf-8"))["characters"])
out=generate(m,a.seed,c2i,i2c,a.length,a.temperature)
print(out)
open(f"outputs/generated_{a.model}.txt","w",encoding="utf-8").write(out)
