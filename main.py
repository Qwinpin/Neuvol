import pandas as pd

import architecture
import evaluation
import data_processing


def main():
    df = pd.read_csv('example.csv')
    x_tmp = df.x.astype(str).tolist()
    y_tmp = df.y.astype(int).tolist()

    # Create objects
    ind = architecture.Individ(stage=1, data_type='text', task_type='classification', classes=3, parents=None)
    ev = evaluation.Evaluator(kfold_number=5, device='cpu')
    data = data_processing.Data(x_raw=x_tmp, y_raw=y_tmp, data_type='text', task_type='classification')

    # Set evaluation parameters
    ev.set_kfold_number(value=4)
    ev.set_verbose(level=0)

    # Show architecture
    print(ind.get_schema())

    # Random mutation
    print('\n\nMutation\n\n')
    ind.mutation(stage=2)

    # Show again
    print(ind.get_schema())

    # Create train data. BE CAREFUL - dont create data before mutation step
    x, y = data.process_data(data_processing=ind.get_data_processing())

    # Show his story and name
    print(ind.history, ind.name)

    # Train this model
    result = ev.train(network=ind, x=x, y=y)

    # Show result as AUC score (default). One value for each class
    print('AUC: ', result)

if __name__ == "__main__":
    main()