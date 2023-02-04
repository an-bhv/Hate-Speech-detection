from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

import pickle

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def read_data(path):
  dataset_df = pd.read_csv(path, names=['text', 'hate'])
  print(dataset_df)
  return dataset_df

dataset_df = read_data('preprocessed.txt')

dataset_df = dataset_df[dataset_df['hate'].notna()]
dataset_df = dataset_df[dataset_df['text'].notna()]
# dataset_df['hate'] = pd.to_numeric(dataset_df['hate'])
train_df, eval_df= train_test_split(dataset_df, test_size=0.15,random_state=42)
print(train_df['hate'])
model_args = ClassificationArgs(num_train_epochs=4, overwrite_output_dir = True)

model = ClassificationModel(
    "bert", "bert-base-multilingual-cased", args=model_args, use_cuda = True
)

model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(result, model_outputs, wrong_predictions)

pickle.dump(model, open("model" + '.pkl', 'wb'))

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def read_data(path):
  dataset_df = pd.read_csv(path, names=['text', 'hate'])
  print(dataset_df)
  return dataset_df

dataset_df = read_data('preprocessed.txt')

dataset_df = dataset_df[dataset_df['hate'].notna()]
dataset_df = dataset_df[dataset_df['text'].notna()]
# dataset_df['hate'] = pd.to_numeric(dataset_df['hate'])
train_df, eval_df= train_test_split(dataset_df, test_size=0.15,random_state=42)
print(train_df['hate'])
model_args = ClassificationArgs(num_train_epochs=4, overwrite_output_dir = True)

model = ClassificationModel(
    "distilbert", "distilbert-base-uncased", args=model_args, use_cuda = True
)

model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(result, model_outputs, wrong_predictions)

pickle.dump(model, open("model" + '.pkl', 'wb'))

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def read_data(path):
  dataset_df = pd.read_csv(path, names=['text', 'hate'])
  print(dataset_df)
  return dataset_df

dataset_df = read_data('preprocessed.txt')

dataset_df = dataset_df[dataset_df['hate'].notna()]
dataset_df = dataset_df[dataset_df['text'].notna()]
# dataset_df['hate'] = pd.to_numeric(dataset_df['hate'])
train_df, eval_df= train_test_split(dataset_df, test_size=0.15,random_state=42)
print(train_df['hate'])
model_args = ClassificationArgs(num_train_epochs=4, overwrite_output_dir = True)

model = ClassificationModel(
    "albert", "ai4bharat/indic-bert", args=model_args, use_cuda = True
)

model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(result, model_outputs, wrong_predictions)

pickle.dump(model, open("model" + '.pkl', 'wb'))



logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def read_data(path):
  dataset_df = pd.read_csv(path, names=['text', 'hate'])
  print(dataset_df)
  return dataset_df

dataset_df = read_data('preprocessed.txt')

dataset_df = dataset_df[dataset_df['hate'].notna()]
dataset_df = dataset_df[dataset_df['text'].notna()]
# dataset_df['hate'] = pd.to_numeric(dataset_df['hate'])
train_df, eval_df= train_test_split(dataset_df, test_size=0.15,random_state=42)
print(train_df['hate'])
model_args = ClassificationArgs(num_train_epochs=3, overwrite_output_dir = True)

model = ClassificationModel(
    "bert", "bert-base-uncased", args=model_args, use_cuda = True
)

model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(result, model_outputs, wrong_predictions)

pickle.dump(model, open("model" + '.pkl', 'wb'))
