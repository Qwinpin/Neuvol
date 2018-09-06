# ![logo](logo.png)Neuvol

*Genetic algorithm to find the best neural network architecture with Keras*

Neuvol is a genetic algorithm API for neural networks based on Keras. The main idea is to work with data only, without direct architecture construction (but it is still possible).

##### Grow your own neural network!

- Data in -> Neural Network Architecture out
- ~~Set your assumptions about architecture, it will be taken into account!~~ (not yet)
- A large number of allowed layers types

### Features:

- Supported data types: texts, images
- CNN, Dense, LSTM, etc layers are available
- Dropout and reshape sub-layers (Flattern) are available too

### How-to-use

Short example of usage pipeline.

- IMDB data:

  ```python
  from keras.datasets import imdb
  
  import neuvol
  
  
  def main():
      (x_train, y_train), (x_test, y_test) = imdb.load_data(
          path="imdb.npz",
          num_words=30000,
          skip_top=0,
          maxlen=100,
          seed=113,
          start_char=1,
          oov_char=2,
          index_from=3)
  
      evaluator = neuvol.Evaluator(x_train, y_train, kfold_number=5)
      mutator = neuvol.Mutator()
  
      evaluator.create_tokens = False
      evaluator.fitness_measure = 'f1'
      options = {'classes': 2, 'shape': (100,), 'depth': 4}
  
      wop = neuvol.evolution.Evolution(
                                      stages=10,
                                      population_size=10,
                                      evaluator=evaluator,
                                      mutator=mutator,
                                      data_type='text',
                                      task_type='classification',
                                      active_distribution=True,
                                      freeze=None,
                                      **options)
      wop.cultivate()
  
      for individ in wop.population_raw_individ:
          print('Architecture: \n')
          print(individ.schema)
          print('\nScore: ', individ.result)
  
  
  if __name__ == "__main__":
      main()
  
  ```



- Cifar10 data:

  ```python
  from keras.datasets import cifar10
  
  import neuvol
  
  
  def main():
      (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  
      evaluator = neuvol.Evaluator(x_train, y_train, kfold_number=5)
      mutator = neuvol.Mutator()
  
      evaluator.create_tokens = False
      evaluator.fitness_measure = 'AUC'
      options = {'classes': 10, 'shape': (32, 32, 3,), 'depth': 4}
  
      wop = neuvol.evolution.Evolution(
                                      stages=10,
                                      population_size=10,
                                      evaluator=evaluator,
                                      mutator=mutator,
                                      data_type='image',
                                      task_type='classification',
                                      active_distribution=False,
                                      freeze=None,
                                      **options)
      wop.cultivate()
  
      for individ in wop.population_raw_individ:
          print('Architecture: \n')
          print(individ.schema)
          print('\nScore: ', individ.result)
  
  
  if __name__ == "__main__":
      main()
  
  ```

  **Important note: you should set shape of data in option dictionary.**

  Also, you can use GPU for calculation. In order to do that add:

  ```python
  evaluator.device = 'gpu'
  ```


### TODO

- [x] Architectures distribution generation
- [x] Images support
- [ ] Regression models
- [ ] Generative networks (??? almost impossible)
- [x] More available layers
- [ ] More options (assumptions)
- [x] Logo
- [ ] Visualization
