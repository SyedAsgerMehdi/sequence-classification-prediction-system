@echo off
echo Installing requirements...
python -m pip install -r requirements.txt
echo.
echo Training RNN, LSTM and GRU...
python train.py --model all
echo.
echo Starting project dashboard...
streamlit run app.py
pause
