from gbdt.model import GBDT
from gbdt.data import DataSet


model = GBDT(tree_depth=3, learning_rate=0.01, max_iter=2000)
dataset = DataSet('data/ages.csv', 'Age')
model.fit(dataset)
x = {'LikesGardening' : False, 'PlaysVideoGames' : True, 'LikesHats' : False}
print model.predict(x)