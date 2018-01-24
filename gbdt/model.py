from gbdt.tree import buildDecisionTree


class GBDT:
    '''A GBDT for regression.'''

    def __init__(self, max_iter=100, learning_rate=0.05, tree_depth=6, loss_function='MSE'):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tree_depth = tree_depth
        self.loss = SquareError() if loss_function=='MSE' else AbsoluteError()
        self.trees = []
        self.F0 = 0

    def fit(self, dataset):
        self.F0 = self.loss.compute_F0(dataset)
        F = {index: self.F0 for index in dataset.indexes()}  # F is the up-to-date result, using index as key
        for i in range(0, self.max_iter):
            residual = self.loss.compute_residual(dataset, F)
            tree = [None]
            buildDecisionTree(dataset, residual, 0, self.tree_depth, tree)
            self.trees.append(tree[0])
            self.loss.update_F(F, tree[0], dataset, self.learning_rate)

    def predict(self, x):
        F = self.F0
        for tree in self.trees:
            F += self.learning_rate * tree.predict(x)
        return F

class LossFunction:
    '''The loss function in the GBDT model. We applied two loss function, square error and absolute error.'''

    def compute_residual(self, dataset, F):
        '''Return a residual for individual entry in the training dataset.

        Args:
            dataset: A dataset containing indexes, xs, and ys.
            F: The up-to-date result of our GBDT for corresponding entries.

        Returns:
            A dict mapping ids to the corresponding residuals.
        '''
        return 42

    def compute_fit(self, dataset, indexes, F):
        '''Given the loss function, and the ys, compute the value that can minimize the loss function.

        Args:
            dataset: A dataset containing indexes and ys.
            indexes: the indexes of entries in the dataset that participate the minimization.
        '''
        return 42

    def compute_F0(self, dataset):
        return self.compute_fit(dataset, dataset.indexes(), None)

    def update_F(self, F, tree, dataset, learning_rate):
        for leaf in tree.leaves:
            indexes = leaf.subset.indexes()  # get the indexes of one region
            gamma = self.compute_fit(dataset, indexes, F)
            leaf.set_gamma(gamma)
            for index in indexes:
                F[index] += gamma * learning_rate


class SquareError(LossFunction):
    def compute_residual(self, dataset, F):
        residual = {}
        for index in F:
            residual[index] = dataset[index][dataset.y] - F[index]

        return residual

    def compute_fit(self, dataset, indexes, F):
        # if F is None, compute F0
        # if F is not None, compute gamma
        ys = [dataset[index][dataset.y] for index in indexes]
        if F == None:
            return sum(ys) / len(ys)
        else:
            Fs = [F[index] for index in indexes]
            return (sum(ys) - sum(Fs)) / len(ys)


class AbsoluteError(LossFunction):
    def compute_fit(self, dataset, indexes, F):
        return 42