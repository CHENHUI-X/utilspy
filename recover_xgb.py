# Recover tree structure from json file of XGBoost model and using it to predict instance .
import xgboost
from xgboost import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import json
import matplotlib.pyplot as plt
import numpy as np
## Load data : mutil class
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# # Load data : binary class
X, y = make_classification(n_samples=100, n_features=6, n_classes= 2, n_clusters_per_class=1 ,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = xgboost.XGBClassifier(
    n_estimators=2,learning_rate=0.1)
model.fit(X_train, y_train)

# 查看树的结构,从根节点,层次遍历,索引为0~n
plot_tree(model.get_booster(), num_trees=0)
plt.show()

# Save model as JSON file
model.save_model('model.json') # which is has scaled the leaf values , so that you can sum up leaf values as final output simply .

with open('model.json', 'r') as f:
    loaded_model = json.load(f)

# print(loaded_model)
class Mypredict:
    """
        Just for binary classification problem
    """
    def __init__(self, json_model_file):
        print("This class just implement for binary classification problem !")
        #  json model file
        self.json_model_file = json_model_file
        # load model from json file
        with open(self.json_model_file, 'r') as f:
            self.loaded_model = json.load(f)
        # Load num of estimators
        self.sklearn_info_str = self.loaded_model['learner']['attributes']['scikit_learn']
        self.sklearn_info = json.loads(self.sklearn_info_str)
         # Get num of class
        self.num_class  = self.sklearn_info['n_classes_']
        # Get num of estimators
        self.num_estimators = self.sklearn_info['n_estimators']
        # Load num of trees
        self.num_trees = eval(
            self.loaded_model['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'])
        '''
            Note : num_trees is not the same as num_estimators
                    when go into mutil classification problem
                    num_trees = num_estimators * num_class
        '''
        # Get trees infos : a list for trees
        self.tree_infos = self.loaded_model['learner']['gradient_booster']['model']['trees']

    def predict(self, X_test):
        return self._mypredict( X_test)

    def _mypredict(self,X_test):
        if self.num_class > 2:
            #  mutil class
            raise NotImplementedError('Mutil class not implemented !')
        else:
            print(f'Num of class is {self.num_class}')
            return self._binary_class_predict(X_test)
    def _binary_class_predict(self,X_test):
        y_pred = []
        for instance in X_test:
            y_pred.append(self._binary_class_predict_instance(instance))
        return y_pred
    def _binary_class_predict_instance(self,instance):
        leaf_values = []
        for tree in self.tree_infos:
            left_children = tree['left_children']
            right_children = tree['right_children']
            split_feature = tree['split_indices']
            split_condition = tree['split_conditions']
            node_index = 0
            while left_children[node_index] != -1 and right_children[node_index] != -1:
                if instance[split_feature[node_index]] <= split_condition[node_index]:
                    node_index = left_children[node_index]
                else:
                    node_index = right_children[node_index]
            # arrive at  leaf node
            leaf_values.append(split_condition[node_index])

        pro = 1 / (1 + np.exp(-np.sum(leaf_values)))
        return  [pro,1-pro]
        # return 0 if sum(leaf_values) > 0 else 1 # It's opposite of the usual implementation

mypredict = Mypredict('model.json')
y_mypred = mypredict.predict(X_test) # Implement by manual
print('---------------手动实现---------------')
print(y_mypred)
print('---------------官方实现---------------')
y_pred = model.predict_proba(X_test) # Implement by official
print(y_pred)
print('---------------实际结果---------------')
print(y_test)
