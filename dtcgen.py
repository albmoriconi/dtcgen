import csv

import click
import graphviz
import numpy as np
from nyoka import skl_to_pmml
from sklearn import tree, pipeline, ensemble, model_selection, datasets


@click.group()
@click.option('-d', '--max-depth', default=0, show_default=True, help='Maximum tree depth, 0 for none')
@click.option('-e', '--n-estimators', default=1, show_default=True, help='Number of estimators')
@click.option('-o', '--out-file', default='out.pmml', show_default=True, help='Name of the output file')
@click.option('-t', '--train-size', default=0.8, show_default=True, help='Fraction of the dataset to use for model training')
@click.pass_context
def main(ctx, max_depth, n_estimators, out_file, train_size):
    """Decision tree ensembles generator."""
    ctx.obj['MAX_DEPTH'] = max_depth if max_depth != 0 else None
    ctx.obj['N_ESTIMATORS'] = n_estimators
    ctx.obj['OUT_FILE'] = out_file
    ctx.obj['TRAIN_SIZE'] = train_size


@main.command('random')
@click.option('-c', '--n-classes', type=int, required=True, help='Number of classes')
@click.option('-f', '--n-features', type=int, required=True, help='Number of features')
@click.option('-s', '--n-samples', type=int, required=True, help='Number of samples')
@click.pass_context
def random_command(ctx, n_classes, n_features, n_samples):
    """Generate a random decision tree ensemble."""
    max_depth = ctx.obj['MAX_DEPTH']
    n_estimators = ctx.obj['N_ESTIMATORS']
    out_file = ctx.obj['OUT_FILE']
    train_size = ctx.obj['TRAIN_SIZE']

    X, Y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features,
        n_repeated=0,
        n_redundant=0,
        n_clusters_per_class=1
    )
    X_train, X_test, Y_train, Y_test = train_split(X, Y, train_size, out_file)

    clf = dtcgen(X_train, Y_train, max_depth, n_estimators, out_file)
    print_accuracy(clf, X_test, Y_test)


@main.command('csv')
@click.argument('infile', type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def csv_command(ctx, infile):
    """Generate a decision tree ensemble from CSV data.

    INFILE is the name of the input data."""
    max_depth = ctx.obj['MAX_DEPTH']
    n_estimators = ctx.obj['N_ESTIMATORS']
    out_file = ctx.obj['OUT_FILE']
    train_size = ctx.obj['TRAIN_SIZE']

    X, Y = read_csv(infile)
    X_train, X_test, Y_train, Y_test = train_split(X, Y, train_size, out_file, infile)

    clf = dtcgen(X_train, Y_train, max_depth, n_estimators, out_file)
    print_accuracy(clf, X_test, Y_test)


def train_split(X, Y, train_size, out_file, infile=None):
    if train_size < 1.0:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=train_size)
        XY_train = [np.append(X_train[i], Y_train[i]) for i in range(len(X_train))]
        XY_test = [np.append(X_test[i], Y_test[i]) for i in range(len(X_test))]
        write_csv(XY_train, f'{out_file}_train.csv')
        write_csv(XY_test, f'{out_file}_test.csv')
    else:
        X_train = X
        Y_train = Y
        if infile is not None:
            X_test, Y_test = read_csv(infile.replace('_train.csv', '_test.csv'))
        else:
            X_test = []
            Y_test = []

    return X_train, X_test, Y_train, Y_test


def dtcgen(X_train, Y_train, max_depth, n_estimators, out_file):
    """Generation function for decision tree ensembles."""
    features_names = [f'feat_{i}' for i in range(len(X_train[0]))]

    if n_estimators <= 1:
        clf = tree.DecisionTreeClassifier(max_depth=max_depth)
        clf = clf.fit(X_train, Y_train)
        draw_graph(clf, out_file, None)
    else:
        clf = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf = clf.fit(list(X_train), list(Y_train))

        for i in range(len(clf.estimators_)):
            draw_graph(clf.estimators_[i], out_file, i)

    pipe = pipeline.Pipeline([('clf', clf)])
    skl_to_pmml(pipeline=pipe, col_names=features_names, pmml_f_name=out_file)

    return clf


def print_accuracy(clf, X_test, Y_test):
    """Print classifier accuracy."""
    c_ans = 0
    for pred, ans in zip(clf.predict(X_test), Y_test):
        if pred == ans:
            c_ans += 1
    print(f"Classifier accuracy: {c_ans / len(Y_test)}")


def draw_graph(clf, out_file, n):
    """Export graph for classifier."""
    dot_data = tree.export_graphviz(clf, filled=True, rounded=True) 
    graph = graphviz.Source(dot_data)
    filename = f'{out_file}.gv' if n is None else f'{out_file}_{n}.gv'
    graph.render(directory=f'{out_file}_export', filename=filename)


def read_csv(filename):
    """Read CSV dataset from file."""
    X = []
    Y = []
    with open(filename) as csv_file:
        csv_data = csv.reader(csv_file)
        for row in csv_data:
            X.append(row[:-1])
            Y.append(row[-1])
    return X, Y


def write_csv(data, filename):
    """Write CSV dataset to file."""
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)


if __name__ == '__main__':
    main(obj={})
