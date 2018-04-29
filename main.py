import pandas as pd

import architecture
import evaluation
import data_processing


def main():
    df = pd.read_csv('example.csv')
    x_tmp = df.x.astype(str).tolist()
    y_tmp = df.y.astype(int).tolist()

    print(len(x_tmp), len(y_tmp))
    # Create objects
    options = {'classes': 3}
    ind = architecture.Individ(stage=1, data_type='text', task_type='classification', parents=None, **options)
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
    result = ev.fit(network=ind)

    # Show result as AUC score (default). One value for each class
    print('AUC: ', result)

if __name__ == "__main__":
    main()
