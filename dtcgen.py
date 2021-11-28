import csv
import random
import sys

import click
import graphviz
from nyoka import skl_to_pmml
from sklearn import tree, pipeline, ensemble, model_selection


@click.group()
@click.option('-d', '--max-depth', default=None, show_default=True, help='Maximum tree depth')
@click.option('-e', '--n-estimators', default=1, show_default=True, help='Number of estimators')
@click.option('-o', '--out-file', default='out.pmml', show_default=True, help='Name of the output file')
@click.pass_context
def main(ctx, max_depth, n_estimators, out_file):
    '''Decision tree ensembles generator.'''
    ctx.obj['MAX_DEPTH'] = max_depth
    ctx.obj['N_ESTIMATORS'] = n_estimators
    ctx.obj['OUT_FILE'] = out_file


@main.command('random')
@click.option('-c', '--n-classes', type=int, required=True, help='Number of classes')
@click.option('-f', '--n-features', type=int, required=True, help='Number of features')
@click.option('-s', '--n-samples', type=int, required=True, help='Number of samples')
@click.pass_context
def random_command(ctx, n_classes, n_features, n_samples):
    '''Generate a random decision tree ensemble.'''
    max_depth = ctx.obj['MAX_DEPTH']
    n_estimators = ctx.obj['N_ESTIMATORS']
    out_file = ctx.obj['OUT_FILE']

    X = [[random.random() for _ in range(n_features)] for _ in range(n_samples)]   
    Y = random.choices(range(n_classes), k=n_samples)

    dtcgen(X, Y, max_depth, n_estimators, out_file)


@main.command('csv')
@click.argument('infile', type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def csv_command(ctx, infile):
    '''Generate a decision tree ensemble from CSV data.

    INFILE is the name of the input data.'''
    max_depth = ctx.obj['MAX_DEPTH']
    n_estimators = ctx.obj['N_ESTIMATORS']
    out_file = ctx.obj['OUT_FILE']

    X = []
    Y = []

    with open(infile) as csv_file:
        csv_data = csv.reader(csv_file)
        for row in csv_data:
            X.append(row[:-1])
            Y.append(row[-1])

    dtcgen(X, Y, max_depth, n_estimators, out_file)


def dtcgen(X, Y, max_depth, n_estimators, out_file):
    '''Generation function for decision tree ensembles.'''
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8)
    features_names = [f'feat_{i}' for i in range(len(X[0]))]

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

    XY_train = [X_train[i] + [Y_train[i]] for i in range(len(X_train))]
    XY_test = [X_test[i] + [Y_test[i]] for i in range(len(X_test))]

    write_csv(XY_train, f'{out_file}_train.csv')
    write_csv(XY_test, f'{out_file}_test.csv')


def draw_graph(clf, out_file, n):
    '''Export graph for classifier.'''
    dot_data = tree.export_graphviz(clf, filled=True, rounded=True) 
    graph = graphviz.Source(dot_data)
    filename = f'{out_file}.gv' if n is None else f'{out_file}_{n}.gv'
    graph.render(directory=f'{out_file}_export', filename=filename)


def write_csv(data, filename):
    '''Write CSV dataset to file.'''
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)


if __name__ == '__main__':
    main(obj={})
