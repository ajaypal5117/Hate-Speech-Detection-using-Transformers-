# Hate Speech Detection using Transformers

A transformer-based text classifier that flags hate speech in tweets, written
from scratch in PyTorch - multi-head self-attention and feed-forward blocks
implemented directly rather than pulled from a pretrained model.

Dataset: [Twitter Hate Speech](https://www.kaggle.com/vkrahul/twitter-hate-speech?select=train_E6oV3lV.csv) (Kaggle), 31,962 labelled posts.

## Results

Trained for 15 epochs on a balanced subset:

| Split | Accuracy |
|---|---|
| Training | 88.8% |
| Validation | 80.5% |

Validation set, 898 posts (balanced):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not hate speech | 0.79 | 0.83 | 0.81 |
| Hate speech | 0.82 | 0.78 | 0.80 |

Held-out test set, 31,962 posts (natural 13:1 imbalance):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Not hate speech | 0.98 | 0.79 | 0.87 | 29,720 |
| Hate speech | 0.20 | 0.76 | 0.32 | 2,242 |

### Reading those two tables together

The validation numbers and the test numbers describe the same model on
differently distributed data, and the gap between them is the most interesting
thing in this project.

Training used undersampling to balance the classes, so the model learned on
roughly equal numbers of each. On balanced validation data it performs
consistently - around 0.80 on both precision and recall for hate speech.

The real test set is not balanced: only 7% of posts are hate speech. Recall
holds up at 0.76, so the model still catches roughly three quarters of actual
hate speech. But precision collapses to 0.20, because a false-positive rate
that looks small against 29,720 negatives produces far more false alarms than
there are true positives. Four out of five flags are wrong.

This is the expected consequence of training on a balanced subset and
evaluating on the real distribution. It is not a bug, but it does mean the
model as trained is unsuitable for automatic moderation - it would be usable as
a first-pass filter feeding human review, where recall matters more than
precision.

Class weighting in the loss function, rather than undersampling the data, would
be the obvious next thing to try.

### Overfitting

Validation accuracy peaked around epoch 9 at 82.2% and then drifted down to
80.5% by epoch 15, while training accuracy kept climbing to 88.8%. The model
starts overfitting roughly two thirds of the way through training. Early
stopping on validation accuracy would have kept the better checkpoint; the full
per-epoch log is in the notebook.

## Architecture

Implemented from scratch rather than fine-tuning a pretrained model, since the
point was to understand the mechanism:

- **Multi-head self-attention.** Queries, keys and values are projected, split
  across heads, and the per-head attention scores computed with `torch.bmm` -
  batched matrix multiplication, so all heads in a batch are handled in one
  parallel operation rather than looped.
- **Feed-forward block** after attention, with residual connections and layer
  normalisation.
- **Global mean pooling** over the sequence, then a linear layer to two classes.

## Pipeline

1. **Download** the dataset from Kaggle via `opendatasets`.
2. **Clean** tweets - regex and BeautifulSoup to strip markup and handles,
   normalise slang, drop irrelevant tokens.
3. **Explore** with wordclouds and class-distribution plots.
4. **Balance** the training data by undersampling the majority class.
5. **Tokenize** with spaCy and build vocabularies with torchtext, batch-first.
6. **Train** the transformer, tracking training and validation accuracy per
   epoch.
7. **Evaluate** with a confusion matrix and classification report on both the
   balanced validation split and the full imbalanced test set.

## Running it

```bash
pip install torch torchtext spacy pandas numpy matplotlib seaborn wordcloud opendatasets
python -m spacy download en_core_web_sm
```

Then run `Toxic tweets.ipynb` top to bottom. The notebook downloads the dataset
itself; you will need Kaggle API credentials the first time.

Note that `torchtext` was deprecated in 2024 and its vocab API has since
changed. The notebook was written against the older interface.

## Known issues

- Precision on the hate-speech class is 0.20 on the natural distribution, for
  the reason set out above. Class-weighted loss is the fix worth trying.
- No early stopping, so the saved weights are from epoch 15 rather than the
  better epoch 9 checkpoint.
- The slang dictionary is hand-built and incomplete; it needs extending and the
  model retraining as usage shifts.
- Single train/validation split rather than cross-validation, so the accuracy
  figures carry more variance than the decimal places suggest.
- Trained weights are not committed - rerun the notebook to reproduce.

## References

- [Attention Is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Peter Bloem - Transformers from scratch](http://peterbloem.nl/blog/transformers)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

## Author

Pal Ajay Ramsagar - github.com/ajaypal5117
