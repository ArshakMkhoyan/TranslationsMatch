# TranslationsMatch
ML model to match company's Russian name with its English one

## Dataset

Dataset contains 3 columns: company name in russian, company name in english, and corresponding boolean variable indicating if these names are for the same company. 

## Pipeline

- First, I process english and russian sentences and transform them to embeddings using pretrained language model [LaBSE](https://www.sbert.net/docs/pretrained_models.html). The embeddings are being saved to latter fit to the model by batches. **SaveProcessedData** notebook contains the described logic in code.

- Then, I fit classification model (NN) with concatenated embeddings of russian and english names as input and boolean feature as output. **FinalPipeline** notebook contains final training code.

## Results

I achived 0.92 F1 score.