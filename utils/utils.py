from tensorflow.keras import layers, Sequential
from tensorflow import keras

from tqdm import tqdm_notebook
from glob import glob
import pandas as pd
import numpy as np
import os


def get_embeddings_and_save(data: pd.DataFrame, embed_model, dir_name: str, process_size: int=None) -> None:
  """Get embeddings for ru_name and eng_name, concat them and save by parts. Utilises GPU for faster computation"""
  if process_size is None:
    process_size = data.shape[0]

  n_iters = (np.ceil(len(data) / float(process_size))).astype(np.int)

  for n_iter in tqdm_notebook(range(n_iters)):
    data_process = data[n_iter * process_size : (n_iter + 1) * process_size]
    
    eng_embedings = embed_model.encode(data_process['eng_name'].to_list(), show_progress_bar=True,
                                       batch_size=256, device='cuda:0', num_workers=4)
    ru_embedings = embed_model.encode(data_process['ru_name'].to_list(), show_progress_bar=True,
                                      batch_size=256, device='cuda:0', num_workers=4)
    
    X_process = np.concatenate((ru_embedings, eng_embedings), axis=1)
    np.save(f'processed_data/{dir_name}/X_processed_{n_iter}.npy', X_process)

    if dir_name != 'test':
      y_process = data_process['answer'].astype(np.bool)
      np.save(f'processed_data/{dir_name}/y_processed_{n_iter}.npy', y_process)


def init_model(input_shape: tuple) -> Sequential:
  """Model initialisation"""
  model = keras.Sequential(
      [
          layers.Dense(128, input_shape=input_shape , activation="relu", name="Layer1"),
          layers.Dense(32, activation="relu", name="Layer2"),
          layers.Dense(1, activation='sigmoid' ,name="SigmoidLayer"),
      ]
  )

  model.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()],
  )
  return model


def get_model(models_train_dir: str) -> Sequential:
  """Initialize model or get already trained model if some epoches were passed before"""
  n_epochs_trained = len(os.listdir(models_train_dir))

  if n_epochs_trained == 0:
    # Initialize model
    model = init_model(input_shape=(1536,))
    print('Initialized new model.')
  else:
    # Load already trained model
    last_model_path = glob(f'{models_train_dir}/model_epoch-{n_epochs_trained}*')[0]
    model = keras.models.load_model(last_model_path)
    print(f"Loaded model: {last_model_path}")
  return model


def train_iter(model: Sequential, n_iter: int) -> Sequential:
  """Train only on part of data which can be loaded to the memmory"""
  train_X = np.load(f'processed_data/full_train/X_processed_{n_iter}.npy')
  train_y = np.load(f'processed_data/full_train/y_processed_{n_iter}.npy')

  model.fit(x=train_X,
            y=train_y,
            batch_size=batch_size,
            epochs=1,
            verbose=1
            )
  return model


def train_epoch(model: Sequential) -> Sequential:
  """Train 1 epoch on full data"""
  for n_iter in range(n_processed):
    model = train_iter(model, n_iter)
  return model