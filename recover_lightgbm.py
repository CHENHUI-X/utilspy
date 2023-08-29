Recover tree structure from json file of LIghtGBM model and using it to predict instance .
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import json
import numpy as np
## Load data : mutil class
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# # Load data : binary class
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, n_clusters_per_class=1 ,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LGBMClassifier(
    n_estimators=2
)
model.fit(X_train, y_train)

# Save model as JSON file
json_model = model.booster_.dump_model()
# which is has scaled the leaf values , so that you can sum up leaf values as final output simply .

with open('model.json', 'w') as f:
    json.dump(json_model, f)

class Mypredict:
    def __init__(self, json_model_file):
        print("This class just implement for binary classification problem !")
        #  json model file
        self.json_model_file = json_model_file
        # load model from json file
        with open(self.json_model_file, 'r') as f:
            self.loaded_model = json.load(f)
        #  Load tree info : a  list
        self.trees_info = self.loaded_model['tree_info']
        #   Load num_tree
        self.num_trees = len(self.trees_info)
        # Load num_class
        self.num_class = self.loaded_model['num_class']
        # Load num_tree_per_iteration
        self.num_tree_per_iteration = self.loaded_model['num_tree_per_iteration']
        # Get num of iteration
        self.num_iteration = int(self.trees_info.__len__() / self.num_tree_per_iteration)
        """
            num_tree_per_iteration  == num_class
            where num_tree_per_iteration is means that the model
            has num_class trees in each iteration.
            For mutil class , the model use one vs rest to train.
        """
    def predict(self, X_test):
        return self._mypredict( X_test)

    def _mypredict(self,X_test):
        if self.num_class > 2:
            #  mutil class
            raise NotImplementedError('Mutil class not implemented')
        else:
            print(f'Num of class is {self.num_class + 1}')
            return self._binary_class_predict(X_test)
    def _binary_class_predict(self,X_test):
        y_pred = []
        for instance in X_test:
            y_pred.append(self._binary_class_predict_instance(instance))
        return y_pred
    def _binary_class_predict_instance(self,instance):
        leaf_values = []
        for tree_info in self.trees_info:
            tree_structure = tree_info['tree_structure']
            while 'leaf_value' not in tree_structure:
                if instance[tree_structure['split_feature']] <= tree_structure['threshold']:
                    tree_structure = tree_structure['left_child']
                else:
                    tree_structure = tree_structure['right_child']
            leaf_values.append(tree_structure['leaf_value'])
        # pro = 1 / (1 + np.exp(-np.sum(leaf_values)))
        return 1 if sum(leaf_values) > 0 else 0


mypredict = Mypredict('model.json')
y_mypred = mypredict.predict(X_test) # Implement by manual
print('---------------手动实现---------------')
print(y_mypred)
print('---------------官方实现---------------')
y_pred = model.predict(X_test) # Implement by official
print(y_pred)
print('---------------实际结果---------------')
print(y_test)
