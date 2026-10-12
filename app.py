import os,json,numpy as np,pandas as pd,streamlit as st
from tensorflow.keras.models import load_model
from utils import display_character,predict_next_character

st.set_page_config(page_title="Sequence Prediction System",page_icon="🧠",layout="wide")
st.title("🧠 Sequence Prediction System")
st.caption("Character-level next-character prediction using RNN, LSTM and GRU")

if not os.path.exists("outputs/metrics.csv"):
    st.warning("Models are not trained yet. Run `python train.py --model all` first.")
    st.code("pip install -r requirements.txt\npython train.py --model all")
    st.stop()

metrics=pd.read_csv("outputs/metrics.csv")
required_metrics={"Test Accuracy","Test Loss","Test Perplexity"}
if not required_metrics.issubset(metrics.columns):
    st.warning("Existing metrics use the old validation-only format. Retrain the models to enable test metrics and perplexity.")
    st.code("python train.py --model all")
    st.stop()
with open("outputs/char_to_idx.json",encoding="utf-8") as f: c2i=json.load(f)
with open("outputs/idx_to_char.json",encoding="utf-8") as f: i2c=np.array(json.load(f)["characters"])

tab1,tab2,tab3=st.tabs(["🔮 Prediction","📊 Comparison","📚 About"])

with tab1:
    model_name=st.selectbox("Choose architecture",["RNN","LSTM","GRU"])
    seed=st.text_input("Enter starting sequence","ROMEO: ")
    top_k=st.slider("Number of candidate characters",1,10,5)
    if st.button("Predict next character",type="primary"):
        path=f"outputs/models/{model_name.lower()}.keras"
        model=load_model(path)
        predictions=predict_next_character(model,seed,c2i,i2c,top_k)
        st.metric("Predicted next character",display_character(predictions[0][0]))
        readable_predictions=[(display_character(character),probability) for character,probability in predictions]
        st.dataframe(pd.DataFrame(readable_predictions,columns=["Character","Probability"]),use_container_width=True)

    st.divider()
    st.subheader("Prompt comparison")
    fixed_prompt=st.selectbox(
        "Choose a default prompt",
        ["Romeo", "To be or not to be", "King"]
    )
    custom_prompt=st.text_input("Enter your own prompt")
    prompt=custom_prompt.strip() or fixed_prompt
    comparison_top_k=st.slider("Candidates per architecture",1,5,3)
    if st.button("Compare all architectures"):
        comparison_columns=st.columns(3)
        for column,architecture in zip(comparison_columns,["RNN","LSTM","GRU"]):
            model=load_model(f"outputs/models/{architecture.lower()}.keras")
            predictions=predict_next_character(model,prompt,c2i,i2c,comparison_top_k)
            with column:
                st.markdown(f"**{architecture}**")
                readable_predictions=[(display_character(character),probability) for character,probability in predictions]
                st.dataframe(pd.DataFrame(readable_predictions,columns=["Character","Probability"]),hide_index=True)
        st.caption(
            "Each model receives the same prompt. Probabilities are measured on the next-character task; "
            "they are not random sentence samples."
        )

with tab2:
    st.subheader("Model Performance")
    st.dataframe(metrics,use_container_width=True)
    st.bar_chart(metrics.set_index("Model")[["Validation Accuracy","Test Accuracy"]])
    st.info("Test accuracy, test loss and perplexity are calculated on the held-out final 10% of the corpus.")

    dependency_path="outputs/dependency_metrics.csv"
    if os.path.exists(dependency_path):
        dependency_metrics=pd.read_csv(dependency_path)
        st.subheader("Context-length experiment")
        st.caption(
            "Validation accuracy for the same architectures trained with sequence lengths 10, 60, and 80. "
            "This is a limited comparison, not evidence of a consistent accuracy drop or long-range superiority."
        )
        st.dataframe(dependency_metrics,use_container_width=True)
        chart_data=dependency_metrics.pivot(
            index="Sequence Length",columns="Model",values="Validation Accuracy"
        ).sort_index()
        st.line_chart(chart_data)
    else:
        st.warning(
            "The context-length results are not available yet. Run the experiment in the README "
            "to create outputs/dependency_metrics.csv."
        )

with tab3:
    st.subheader("Project Objective")
    st.write("The system predicts exactly one next character from a fixed-length English text sequence. It does not generate random sentences or translate text.")
    st.markdown("""
**Simple RNN:** baseline recurrent architecture; useful for nearby dependencies but can struggle with long dependencies.

**LSTM:** uses a cell state and gates to control information flow.

**GRU:** uses update and reset gates with a simpler architecture than LSTM.

**Pipeline:** Text → Character Encoding → Sequences → Embedding → Recurrent Layer → Softmax → Next Character
""")
