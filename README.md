Neural Network Tournament Forecaster

Overview

Neural Network Tournament Forecaster is a machine learning framework designed to evaluate, compare, and rank multiple neural network architectures for financial time-series forecasting. Rather than assuming that a single architecture is optimal, the system allows competing neural network designs to participate in a structured tournament where performance on unseen data determines which models advance and which models are eliminated.

The project was inspired by a simple observation: many machine learning experiments focus heavily on training a model but spend insufficient effort on validating whether the model genuinely generalizes to new data. Financial forecasting is particularly vulnerable to this problem because stock-market datasets contain noise, regime changes, and hidden patterns that can easily cause models to overfit.

To address this challenge, the project introduces a tournament-based evaluation system where multiple architectures compete across separate datasets, ultimately producing a champion model based on objective out-of-sample performance.

---

Project Objective

The objective of the system is to forecast future market behavior using historical price information while simultaneously identifying which neural network architecture performs best under realistic evaluation conditions.

Instead of manually selecting an architecture, the framework trains multiple candidate architectures, evaluates them on unseen datasets, ranks them according to predictive performance, and promotes only the strongest models to a final championship round.

This transforms model selection from a subjective decision into a data-driven competitive process.

---

Dataset Design

The system uses three completely separate datasets:

Training Dataset

The training dataset is used exclusively for model learning.

All candidate architectures are trained using this dataset.

The training data contains historical market information that allows neural networks to discover patterns and relationships within the price series.

Mock Examination Dataset

The mock examination dataset acts as the first independent evaluation stage.

No training occurs on this dataset.

Instead, every trained model is tested against unseen data to determine how well it generalizes beyond the training environment.

The purpose of this stage is to eliminate architectures that appear successful during training but fail to perform consistently on new data.

Final Examination Dataset

The final examination dataset serves as the ultimate evaluation stage.

Only the strongest architectures from the mock examination advance to this phase.

Performance on this dataset determines the final ranking and tournament champion.

Because the final examination data remains unseen throughout the selection process, it provides a more realistic estimate of future performance.

---

Data Preparation Pipeline

The forecasting system uses a rolling-window methodology.

For each training example:

- The previous 90 trading periods are used as model input.
- The following 30 trading periods are used as prediction targets.

This creates a forecasting framework capable of predicting future market behavior using recent historical observations.

The system continuously slides this window across the dataset, generating thousands of training examples.

Normalization Strategy

A key challenge in financial forecasting is that stock prices can exist at very different scales.

A stock trading at 100 and a stock trading at 10,000 may exhibit similar behavior despite having dramatically different absolute prices.

To solve this problem, each sequence is normalized relative to an anchor price.

The anchor price is defined as the final observation within the 90-day lookback window.

All values within the sequence are divided by this anchor value.


Neural Network Tournament System

The heart of the project is the tournament architecture.

Rather than training a single model, the framework evaluates ten competing neural network architectures.

Each architecture contains a different layer configuration and neuron structure.

Examples include:

- Deep compression architectures
- Symmetric expansion-compression architectures
- Progressive narrowing architectures
- Large-capacity dense networks
- Experimental hybrid structures

Each architecture represents a different hypothesis about how information should flow through a neural network.

---

Model Construction

Every candidate architecture is automatically converted into a TensorFlow neural network.

The system uses:

- Dense layers
- ReLU activation functions
- Adam optimization
- Mean Absolute Error (MAE) loss

The final output layer generates a 30-step forecast horizon.

This allows the network to predict future market movements rather than only a single future point.

---

Performance Optimization Through Caching

Training large forecasting systems can be computationally expensive.

To reduce unnecessary processing, the project includes an intelligent caching system.

Once dataset windows are generated:

- Input arrays are saved as NumPy files.
- Target arrays are saved as NumPy files.
- Future executions load directly from cache.

This dramatically reduces preprocessing time and allows experimentation to focus on model evaluation rather than repetitive data preparation.

---

Mock Examination Stage

After training, every architecture enters the mock examination stage.

Each model generates predictions for the unseen mock dataset.

Performance is measured using directional accuracy.

Directional accuracy evaluates whether the model correctly predicts market direction rather than exact prices.

This metric is particularly relevant in financial forecasting because correct directional decisions often matter more than perfect numerical forecasts.

All architectures are ranked according to their mock examination performance.

---

Selection of Finalists

After ranking is complete, the top-performing architectures advance to the final examination stage.

Only the strongest models survive.

This creates a competitive environment similar to a sports tournament where weaker participants are eliminated and only the best performers continue.

The framework automatically exports the finalists and prepares them for championship evaluation.

---

Final Examination Stage

The final examination serves as the most important validation phase.

Each finalist is evaluated using completely unseen market data.

Several performance metrics are calculated:

Directional Accuracy

Measures the percentage of correct market direction predictions.

Mean Absolute Error (MAE)

Measures average prediction error.

Root Mean Squared Error (RMSE)

Measures error magnitude while penalizing larger mistakes more heavily.

Correlation

Measures how closely predicted movements align with actual market behavior.

Together these metrics provide a comprehensive assessment of model quality.

---

Champion Selection

After all finalists complete the final examination, results are ranked according to final performance.

The highest-performing architecture is crowned the tournament champion.

A detailed report card is generated containing:

- Architecture name
- Mock examination accuracy
- Final examination accuracy
- MAE
- RMSE
- Correlation score

The complete leaderboard is also exported for future analysis.

---

Key Machine Learning Concepts Demonstrated

This project demonstrates practical application of:

- Neural Networks
- Deep Learning
- Financial Time-Series Forecasting
- Feature Normalization
- Model Selection
- Architecture Benchmarking
- Generalization Testing
- Overfitting Prevention
- Performance Evaluation
- Experiment Management
- Predictive Analytics
- TensorFlow Development

---

Conclusion

Neural Network Tournament Forecaster was designed as a machine learning experimentation framework focused on objective model evaluation rather than simple model training. By combining multiple architectures, independent validation stages, automated benchmarking, and championship-style selection, the project provides a structured methodology for identifying robust forecasting models capable of performing on unseen financial data.

The project demonstrates how machine learning systems can be evaluated through competition, allowing data-driven evidence rather than assumptions to determine which neural network architecture performs best.


TOURNAMENT REPORTCARD :


MOCK EXAM LEADERBOARD

1. Layer_1 | 57.43%


2. Layer_2 | 57.43%


3. Layer_3 | 57.43%


4. Layer_4 | 57.43%


5. Layer_5 | 57.43%


6. Layer_6 | 57.43%


7. Layer_7 | 57.43%


8. Layer_8 | 57.43%


9. Layer_9 | 57.43%


10. Layer_10 | 57.43%



======================================================================
TOP 5 FINALISTS ADVANCING

1. Layer_1


2. Layer_2


3. Layer_3


4. Layer_4


5. Layer_5



Saved: top_5_finalists.csv - Ready for Final Exam Stage.

======================================================================
FINAL EXAM TOURNAMENT


---

Final Exam Candidate: Layer_1
Mock Accuracy: 57.43%

Final Accuracy: 58.31%


---

Final Exam Candidate: Layer_2
Mock Accuracy: 57.43%

Final Accuracy: 58.31%


---

Final Exam Candidate: Layer_3
Mock Accuracy: 57.43%

Final Accuracy: 58.31%


---

Final Exam Candidate: Layer_4
Mock Accuracy: 57.43%

Final Accuracy: 58.31%


---

Final Exam Candidate: Layer_5
Mock Accuracy: 57.43%

Final Accuracy: 58.31%

================================================================================
FINAL EXAM LEADERBOARD

1. Layer_1 | Mock=57.43% | Final=58.31%


2. Layer_2 | Mock=57.43% | Final=58.31%


3. Layer_3 | Mock=57.43% | Final=58.31%


4. Layer_4 | Mock=57.43% | Final=58.31%


5. Layer_5 | Mock=57.43% | Final=58.31%



================================================================================
🏆 TOURNAMENT CHAMPION REPORT CARD

Champion Architecture: Layer_1
Mock Exam Accuracy:    57.43%
Final Exam Accuracy:   58.31%
MAE:                   0.056879
RMSE:                  0.082989
Correlation:           0.0911

Saved: tournament_final_results.csv

🏆 WINNER: Layer_1 with Final Exam Accuracy 58.31%

