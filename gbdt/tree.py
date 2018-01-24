import sys
from data import DataSet


class DecisionTree:
    def __init__(self, root):
        '''
        Construct a tree.
        '''
        self.root = root
        self.leaves = []

    def add_leaf(self, leaf):
        if not leaf.is_leaf:
            raise TypeError("This node is not leaf.")
        self.leaves.append(leaf)

    def predict(self, x):
        ## x is a dict of features
        return self.root.predict(x)


class DecisionNode:
    def __init__(self, attribute, subset, split_at=None, is_leaf=False):
        self.attribute = attribute
        self.subset = subset
        self.split_at = split_at
        self.is_leaf = is_leaf
        self.left_child = None
        self.right_child = None
        self.gamma = None

        if not self.is_leaf:
            self.left_subset = DataSet(dataset=subset, indexes=[])
            self.right_subset = DataSet(dataset=subset, indexes=[])
            self.compute_left_right_subset()

    def compute_left_right_subset(self):
        for entry in self.subset:
            if entry[entry.keys()[0]][self.attribute] < self.split_at:
                self.left_subset.append(entry)
            else:
                self.right_subset.append(entry)

    def set_gamma(self, gamma):
        if self.is_leaf and self.gamma == None:
            self.gamma = gamma
        else:
            raise TypeError('This is not leaf or this gamma is already set.')

    def predict(self, x):
        if self.is_leaf:
            return self.gamma
        if x[self.attribute] < self.split_at:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)


def SE(vals):
    '''Square Error loss'''
    if len(vals) <= 1:
        return 0
    mean = sum(vals) * 1.0 / len(vals)
    square_errors = [(val - mean) ** 2 for val in vals]
    return sum(square_errors)


def compute_loss(node, targets):
    if node.is_leaf:
        corresponding_targets = []
        for entry in node.subset:
            corresponding_targets.append(targets[entry.keys()[0]])
        return SE(corresponding_targets)
    else:
        corresponding_targets = []
        result = 0
        for entry in node.left_subset:
            corresponding_targets.append(targets[entry.keys()[0]])
        result += SE(corresponding_targets)
        corresponding_targets = []
        for entry in node.right_subset:
            corresponding_targets.append(targets[entry.keys()[0]])
        return result + SE(corresponding_targets)


def get_split_ats(vals):
    ## get an ordered list of split points for decision nodes, based on the input value set
    ## if the value set containing only one value, then return None
    ## if the value set containing only True and False, then return [0.5]
    val_set = set(vals)
    if len(val_set) <= 1:
        return None
    sorted_values = sorted(val_set)
    result = []
    for i in range(len(sorted_values) - 1):
        result.append((sorted_values[i] + sorted_values[i + 1]) / 2.0)

    return result


def buildDecisionTree(subset, targets, cur_lvl, depth, tree):
    '''
    Build a decision tree. Return the tree.
    '''
    if cur_lvl < depth - 1:  # non-leaf
        min_loss = sys.maxint
        min_node = None
        attributes = subset.attributes
        for attribute in attributes:  # loop through all attributes to find the best splitting strategy
            split_ats = get_split_ats(subset.get_vals(attribute))
            if split_ats == None:
                node = DecisionNode(None, subset, is_leaf=True)
                cur_loss = compute_loss(node, targets)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    min_node = node
            else:
                for split_at in split_ats:  # loop through all possible splits to find the best splitting strategy
                    node = DecisionNode(attribute, subset, split_at=split_at)
                    cur_loss = compute_loss(node, targets)
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        min_node = node
        if cur_lvl == 0:  # root
            tree[0] = DecisionTree(min_node)
        if min_node.is_leaf:
            tree[0].add_leaf(min_node)
            return min_node
        else:
            min_node.left_child = buildDecisionTree(min_node.left_subset, targets, cur_lvl + 1, depth, tree)
            min_node.right_child = buildDecisionTree(min_node.right_subset, targets, cur_lvl + 1, depth, tree)
            return min_node
    else:  # leaf
        node = DecisionNode(None, subset, is_leaf=True)
        tree[0].add_leaf(node)
        return node

