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


## References
###### Zhou, Peng, et al. "Attention-based bidirectional long short-term memory networks for relation classification." Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). Vol. 2. 2016.  
###### dos Santos, Cicero, Bing Xiang, and Bowen Zhou. "Classifying Relations by Ranking with Convolutional Neural Networks." Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Vol. 1. 2015.  

## Required Python package
python3
pytorch==0.4.0  
scikit-learn==0.20.1  
networkx==2.2  
spacy==2.0.16  
