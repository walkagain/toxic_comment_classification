### Data Description

 A large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

**load the dataset from [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)**

## File descriptions

- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.
- **sample_submission.csv** - a sample submission file in the correct format
- **test_labels.csv** - labels for the test data; value of `-1` indicates it was not used for scoring; (**Note:** file added after competition close!)

## Requirements

- Python 3.6
- sklearn
- numpy
- pandas

## Train and Evaluation

```
./toxic_comment_classification.py
```

**accuracy results showed like that**

```
predict accuracy of toxic: 0.925
predict accuracy of severe_toxic: 0.993
predict accuracy of obscene: 0.961
predict accuracy of threat: 0.997
predict accuracy of insult: 0.958
predict accuracy of identity_hate: 0.990
avg predict accuracy: 0.971
```

**predict results would be written into a file named 'submission.csv'**

