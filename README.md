# Relation Classification

## Task
SemEval2010_task8
(https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview)

## Data
Generate data for training models. Please set featdict with 'PI': True for baseRNN and attnRNN models.
SemEval2010_task8 has been placed in 'data' directory.
Please set 'embed_file' with your pre-trained word embedding file.


run: python REData.py

## Train model
The hyper-parameters can be set in the main function of 'REOptimize.py'.
Currently, 'baseRNN' and 'attnRNN' is available. Please set 'classification_model' in the 'REOptimize.py'.

run: python REOptimize.py


## Required Python package
pytorch==0.4.0  
scikit-learn==0.20.1  
networkx==2.2  
spacy==2.0.16  
