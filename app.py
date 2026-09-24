import os,json,numpy as np,pandas as pd,streamlit as st
from tensorflow.keras.models import load_model
from utils import generate

st.set_page_config(page_title="Sequence Prediction System",page_icon="🧠",layout="wide")
st.title("🧠 Sequence Prediction System")
st.caption("Character-level next-character prediction using RNN, LSTM and GRU")

if not os.path.exists("outputs/metrics.csv"):
    st.warning("Models are not trained yet. Run `python train.py --model all` first.")
    st.code("pip install -r requirements.txt\npython train.py --model all")
    st.stop()

metrics=pd.read_csv("outputs/metrics.csv")
with open("outputs/char_to_idx.json",encoding="utf-8") as f: c2i=json.load(f)
with open("outputs/idx_to_char.json",encoding="utf-8") as f: i2c=np.array(json.load(f)["characters"])

tab1,tab2,tab3=st.tabs(["🔮 Prediction","📊 Comparison","📚 About"])

with tab1:
    model_name=st.selectbox("Choose architecture",["RNN","LSTM","GRU"])
    seed=st.text_input("Enter starting sequence","ROMEO: ")
    col1,col2=st.columns(2)
    with col1: length=st.slider("Characters to generate",50,500,200)
    with col2: temp=st.slider("Temperature",0.2,1.5,0.8,0.1)
    if st.button("Generate Prediction",type="primary"):
        path=f"outputs/models/{model_name.lower()}.keras"
        model=load_model(path)
        result=generate(model,seed,c2i,i2c,length,temp)
        st.subheader("Generated Sequence")
        st.text_area("Output",result,height=300)

with tab2:
    st.subheader("Model Performance")
    st.dataframe(metrics,use_container_width=True)
    st.bar_chart(metrics.set_index("Model")["Validation Accuracy"])
    st.info("Compare validation accuracy, loss, training time and parameter count. Use the measured results in the final report.")

with tab3:
    st.subheader("Project Objective")
    st.write("The system predicts the next character from a fixed-length sequence and investigates how Simple RNN, LSTM and GRU handle short-term and long-term dependencies.")
    st.markdown("""
**Simple RNN:** baseline recurrent architecture; useful for nearby dependencies but can struggle with long dependencies.

**LSTM:** uses a cell state and gates to control information flow.

**GRU:** uses update and reset gates with a simpler architecture than LSTM.

**Pipeline:** Text → Character Encoding → Sequences → Embedding → Recurrent Layer → Softmax → Next Character
""")
