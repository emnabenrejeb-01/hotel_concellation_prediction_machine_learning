# Hotel Booking Cancellation Prediction (Machine Learning)

Predicting whether a hotel booking will be canceled, using classification models in Python. Group project by Syrine Dhouib and Emna Ben Rejeb.

## Business problem
Cancellations cost hotels revenue and make occupancy hard to plan. This project
identifies which bookings are most at risk of cancellation, so the hotel can act
early (deposit policy, follow-up, overbooking decisions).

## Data
- **Source:** Kaggle, Hotel Booking Demand
- **Size:** about 119,000 bookings (City Hotel and Resort Hotel, 2015 to 2017)
- **Target:** `is_canceled` (37.04% of bookings are canceled)
- **Features:** 13 selected variables (7 categorical, 6 numerical) such as lead_time,
  deposit_type, market_segment, total_of_special_requests, adr, previous_cancellations

## Approach
1. **Exploration:** descriptive statistics, target balance, cancellation rate by feature
2. **Preparation:** no missing values, categorical encoding, features (X) / target (y) split
3. **Feature selection:** low-variance filter, Pearson correlation, ANOVA (numerical),
   chi-square (categorical), and RFE (wrapper method)
4. **Modeling with hyperparameter tuning:** Linear SVM, Decision Tree, Random Forest,
   Naive Bayes, Logistic Regression, Gradient Boosting, ANN (neural network)
5. **Evaluation:** accuracy, F1-score, and confusion matrices

## Key insights
- **Lead time:** bookings made far in advance are more likely to be canceled.
- **Special requests:** guests with more special requests cancel less
  (about 48% cancellation with none, about 5% with five).
- **Market segment:** Online TA is the largest channel, and Groups is the only
  segment with more cancellations than non-cancellations.
- **Deposit type:** non-refundable deposits rank first in the chi-square test.

## Results
| Model | Accuracy | F1-score |
|---|---|---|
| ANN | 0.815 | 0.716 |
| Random Forest | 0.814 | 0.704 |
| Gradient Boosting | 0.811 | 0.702 |
| Decision Tree | 0.803 | 0.696 |
| Logistic Regression | 0.775 | 0.695 |
| Linear SVM | 0.793 | 0.651 |
| Naive Bayes | 0.760 | 0.536 |

The ANN performed best, but Random Forest is almost identical.
The ANN detects about 63% of real cancellations, with about 83% precision
on the bookings it flags.

## Recommendations
- Monitor high lead-time bookings, which carry a higher cancellation risk.
- Give special attention to guests with special requests, who are more likely to stay.
- Use the model's risk score to prioritize follow-up on risky bookings.
- Next step: integrate the predictions into a Power BI dashboard.

## Repository structure
- `Code/`: notebook with the full analysis
- `Presentation/`: PDF presentation of the project

## Tools
Python (pandas, NumPy, scikit-learn, Matplotlib, Seaborn)
