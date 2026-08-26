# Toxic Tweets - Hate Speech Detection using Transformers

This project demonstrates hate speech detection on Twitter data using deep learning and transformer architectures in PyTorch.

## Project Overview

- **Goal:** Classify tweets as hate speech or not.
- **Approach:** Data preprocessing, visualization, and training a transformer-based classifier.
- **Dataset:** [Kaggle - Twitter Hate Speech](https://www.kaggle.com/vkrahul/twitter-hate-speech?select=train_E6oV3lV.csv)

## Steps

1. **Data Download:** Fetch dataset from Kaggle.
2. **Preprocessing:** Clean tweets using regex and BeautifulSoup, normalize slang, and remove irrelevant tokens.
3. **Visualization:** Use wordclouds and bar plots to explore data.
4. **Balancing:** Apply undersampling to address class imbalance.
5. **Feature Engineering:** Tokenize and build vocabularies using torchtext and spaCy.
6. **Model:** Implement a transformer-based neural network for classification.
7. **Training & Validation:** Train the model, evaluate accuracy, and save weights.
8. **Inference:** Predict labels for new tweets and test dataset.
9. **Evaluation:** Use confusion matrix and classification report for metrics.

## Requirements

- Python 3.8+
- PyTorch
- torchtext
- spaCy (`en_core_web_sm`)
- pandas, numpy, matplotlib, seaborn, wordcloud
- opendatasets

Install dependencies:
```bash
pip install torch torchtext spacy pandas numpy matplotlib seaborn wordcloud opendatasets
python -m spacy download en_core_web_sm
```

## Usage

Run the notebook `Toxic tweets.ipynb` step by step.  
Update file paths as needed for your environment.

## Results

- Achieved ~88% training accuracy and ~80% validation accuracy.
- Model can predict hate speech in new tweets and export results.

## References

- [Attention Is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Peter Bloem's Transformer Blog](http://peterbloem.nl/blog/transformers)
- [Harvard NLP - The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

---

**Note:** For best results, keep updating the slang dictionary and retrain the model