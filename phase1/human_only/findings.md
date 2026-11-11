# Findings - Human-only Analysis

We analyzed the model's predictions on **55,914 test recordings** from the WINGBEATS dataset, which contains six mosquito species. We also looked at some of the recordings and their characteristics. Our goal was to understand **where the model makes mistakes and what might be related to them**, rather than only looking at the overall accuracy.

## 1. Dataset and accuracy across species

The six species are not equally represented in the dataset. The model also does not perform equally well on all of them: there is a difference of about **20 percentage points** between the species with the highest and lowest test accuracy.

The two species with the fewest recordings also have the lowest accuracy. This suggests that the number of recordings might matter, but we cannot say that it is the reason for the difference.

## 2. Which species does the model confuse the most?

The biggest problem is **An. arabiensis being predicted as An. gambiae**. This happens in **16.7%** of *An. arabiensis* test recordings. In the opposite direction, *An. gambiae* is predicted as *An. arabiensis* in only **6.3%** of cases. So the confusion is not equally strong in both directions.

Another common mistake is **Ae. albopictus being predicted as Ae. aegypti**, which happens in almost **10%** of cases.

We also grouped mistakes by how closely related the species are. The most common mistakes are between species from the same genus or subfamily. The **eight most common confusion pairs** are within the same subfamily, and the **four most common pairs** are within the same genus. We saw a similar pattern for high-confidence mistakes. This suggests that related species may be harder for the model to distinguish.

## 3. Can the model be confident and still be wrong?

Yes. When we compared correct and incorrect predictions, we saw that **wrong predictions usually have lower confidence**. However, the model can still make mistakes with high confidence.

Across the test set, there were **1,113 incorrect predictions with at least 80% confidence**. Even among these confident mistakes, related species were often confused with each other. So high confidence does not always mean that a prediction is correct.

## 4. Do recording characteristics affect the mistakes?

We focused mainly on *An. arabiensis* recordings that were incorrectly predicted as *An. gambiae*.

### Signal strength

We compared the strength of the recordings using `raw_rms`. Recordings with weaker signals sometimes had more mistakes. For example, the error rate was about **21% in signal-strength group 4**, compared with about **12% in the strongest group**. However, the error rate did not decrease consistently as signal strength increased, so this relationship is not very clear.

### Frequency

We tried to use `spectral_peak_hz` to check whether certain frequencies were more difficult for the model. However, many values were very close to **0 Hz**, which did not seem to represent the mosquito wingbeat frequency. Because of this, we could not make a useful conclusion from these plots.

### Activity near the edges of recordings

We checked whether recordings with strong activity at the beginning or end were more often classified incorrectly. With our highest threshold (**0.9**), the error rate was **18.18%** for recordings with high edge activity and **16.61%** for those with low edge activity.

This difference was small, so we did not find a clear connection between edge activity and the mistakes. Also, strong edge activity does not necessarily mean that a recording was actually cut off.

### Estimated signal-to-noise ratio (SNR)

We wanted to check whether background noise might make classification harder. Each `D_...` folder represents one recording session, so we used the **quietest parts across the whole session** to estimate background noise. We then compared this estimate with the strongest parts of each recording.

This gave one of our clearest results. In the lower estimated-SNR groups (**1-5**), around **18-20%** of *An. arabiensis* recordings were wrongly predicted as *An. gambiae*. In the highest-SNR group (**8**), the error rate was only around **11%**.

This suggests that recordings with lower estimated SNR are harder for the model to classify. However, **SNR is only an estimate**: the quietest parts of a session might still contain mosquito sounds. We therefore cannot say for sure that background noise caused these mistakes.

## 5. Does performance differ between recording sessions?

Yes, **accuracy varies between recording sessions**. Some sessions had noticeably lower accuracy than others, and we found groups of sessions or dates where the model seemed to perform worse.

However, recordings were not spread evenly across dates for all species. Also, some sessions contained only a small number of test recordings, so their accuracy percentages can change a lot because of only one or two mistakes. Because of this, we cannot draw strong conclusions about why certain sessions are harder.

## Overall conclusion

Our human-only analysis showed that the model's mistakes are **not completely random**. It often confuses related mosquito species, especially *An. arabiensis* and *An. gambiae*. Wrong predictions usually have lower confidence, but some are still made with high confidence.

We also found that **signal strength and especially estimated SNR** seem to be related to some mistakes. Edge activity did not show a clear connection, and our frequency measurement was not useful for answering that question. Finally, model accuracy differs across species and recording sessions, although the available data does not tell us exactly why.

These findings help us identify interesting cases to investigate further in **Phase II**, where we can look more closely at what the model uses when making its predictions.
