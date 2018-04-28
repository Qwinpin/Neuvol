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
    ev = evaluation.Evaluator(x_tmp, y_tmp, kfold_number=2, device='cpu', generator=False)

    # Set evaluation parameters
    ev.set_verbose(level=1)

    # Show architecture
    print(ind.get_schema())

    # Random mutation
    print('\n\nMutation\n\n')
    ind.mutation(stage=2)

    # Show again
    print(ind.get_schema())

    # Show his story and name
    print(ind.get_history(), ind.get_name())

    # Train this model
    result = ev.fit_generator(network=ind)

    # Show result as AUC score (default). One value for each class
    print('AUC: ', result)

if __name__ == "__main__":
    main()