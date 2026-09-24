# Sequence Prediction System Using Recurrent Neural Networks

## Abstract
This project develops a character-level sequence prediction system using recurrent neural networks. The system receives a fixed-length sequence of characters and predicts the next character. Simple RNN, LSTM and GRU architectures are trained and compared to investigate their ability to model short-term and long-term dependencies.

## Objectives
1. Build a sequence prediction pipeline.
2. Convert text into fixed-length character sequences.
3. Train Simple RNN, LSTM and GRU models.
4. Compare validation accuracy, loss, parameter count and training time.
5. Generate text from a user-provided seed sequence.

## Dataset
Tiny Shakespeare is used as the default dataset. The dataset contains Shakespearean text and is treated as a character sequence.

## Methodology
Text is converted into integer character IDs. For every window of N characters, the next character becomes the target. The data is split chronologically into training and validation portions.

## Architecture
Embedding → Recurrent Layer → Dropout → Dense Softmax

Three recurrent layers are evaluated:
- SimpleRNN
- LSTM
- GRU

## Short-Term vs Long-Term Dependencies
A Simple RNN maintains a hidden state but can experience vanishing/exploding gradients when dependencies span many time steps. LSTM introduces a cell state and gates to regulate information. GRU uses update and reset gates and has a simpler structure.

## Evaluation
The experiment records:
- Validation accuracy
- Validation loss
- Training time
- Trainable parameter count
- Generated sequence quality

## Results
Run the experiment and copy the values from `outputs/metrics.csv` into this table.

| Architecture | Validation Accuracy | Validation Loss | Training Time | Parameters |
|---|---:|---:|---:|---:|
| Simple RNN | | | | |
| LSTM | | | | |
| GRU | | | | |

## Conclusion
The experiment provides an empirical comparison of recurrent architectures. The final conclusion should be based on the measured results. LSTM and GRU include gating mechanisms designed to preserve useful information over longer dependencies, while Simple RNN provides a simpler baseline.

## Future Scope
- Word-level prediction
- Sentiment classification
- Sensor-based human activity classification
- Time-series forecasting
- Bidirectional RNNs
- Attention and Transformer models
