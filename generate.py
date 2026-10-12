import argparse,json,numpy as np
from tensorflow.keras.models import load_model
from utils import display_character,predict_next_character,set_seed

ap=argparse.ArgumentParser()
ap.add_argument("--model",choices=["rnn","lstm","gru"],default="lstm")
ap.add_argument("--seed",default="ROMEO: ")
ap.add_argument("--top_k",type=int,default=5)
a=ap.parse_args()
set_seed(42)
m=load_model(f"outputs/models/{a.model}.keras")
c2i=json.load(open("outputs/char_to_idx.json",encoding="utf-8"))
i2c=np.array(json.load(open("outputs/idx_to_char.json",encoding="utf-8"))["characters"])
predictions=predict_next_character(m,a.seed,c2i,i2c,max(1,a.top_k))
print(f"Prompt: {a.seed!r}")
print(f"Predicted next character: {display_character(predictions[0][0])}")
print("Top candidates:")
for character,probability in predictions:
	print(f"  {display_character(character)}: {probability:.4f}")
