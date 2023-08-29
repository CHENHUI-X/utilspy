# Using the GA algorithm to optimize parameter of GBDT
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from tqdm import tqdm
# 读取数据
data = load_iris()
X = data.data
y = data.target

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义适应度函数
def fitness_function(params):
    # params 为优化的参数
    gbc = GradientBoostingClassifier(
        n_estimators=int(params[0]),
        learning_rate=params[1],
        max_depth=int(params[2]),
        min_samples_split = int(params[3]),
        min_samples_leaf=int(params[4]),
        subsample=params[5],
        random_state=42)
    # 训练模型
    gbc.fit(X_train, y_train)
    # 在测试集上进行验证
    y_pred = gbc.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 令准确率为fitness
    return accuracy

# 参数范围
param_ranges = [(50, 200), (0.01, 0.5), (1, 10), (2, 20), (1, 10), (0.5, 1.0)]

# 遗传算法
def genetic_algorithm(fitness_function, param_ranges, num_generations, population_size):
    #  个体初始化
    population = []
    for i in tqdm(range(population_size),desc='群体初始化 '):
        individual = []
        for param_range in param_ranges:
            param = np.random.uniform(param_range[0], param_range[1])
            individual.append(param)
        population.append(individual)

    # 进行每轮的迭代
    for i in tqdm(range(num_generations),desc='Gen 迭代 '):
        # 每个个体进行进化
        fitness_scores = []
        for individual in tqdm(population,desc=f'Population 迭代'):
            fitness_scores.append(fitness_function(individual))

        # 选择交叉变异的母体
        parents = []
        for j in range(population_size):
            parent1 = population[np.random.randint(0, population_size)]
            parent2 = population[np.random.randint(0, population_size)]
            if fitness_function(parent1) > fitness_function(parent2):
                parents.append(parent1)
            else:
                parents.append(parent2)

        # 进行交叉和变异
        children = []
        for j in range(population_size):
            parent1 = parents[np.random.randint(0, population_size)]
            parent2 = parents[np.random.randint(0, population_size)]
            child = []
            for k in range(len(param_ranges)):
                # 变异
                if np.random.rand() < 0.5:
                    child.append(parent1[k])
                else:
                    child.append(parent2[k])
                # 变异
                child[k] += np.random.normal(0, 0.1)
                if child[k] < param_ranges[k][0]:
                    child[k] = param_ranges[k][0]
                elif child[k] > param_ranges[k][1]:
                    child[k] = param_ranges[k][1]
            children.append(child)

        # 更新群体
        population = children

    # 寻找最优解
    best_individual = population[0]
    best_fitness = fitness_function(best_individual)
    for individual in population:
        fitness = fitness_function(individual)
        if fitness > best_fitness:
            best_individual = individual
            best_fitness = fitness

    return best_individual, best_fitness

# 运行GA算法
best_params, best_fitness = genetic_algorithm(fitness_function, param_ranges, 5, 5)

# 得到最优参数
best_params = dict(
    [
        ("n_estimators",int(best_params[0])),
        ("learning_rate",best_params[1]),
        ("max_depth",int(best_params[2])),
        ("min_samples_split" , int(best_params[3])),
        ("min_samples_leaf",int(best_params[4]) ),
        ("subsample",best_params[5])
    ]
)

# 使用最优参数进行模型的训练
gbc = GradientBoostingClassifier(
        n_estimators = best_params['n_estimators'],
        learning_rate = best_params['learning_rate'],
        max_depth = best_params['max_depth'],
        min_samples_split = best_params['min_samples_split'],
        min_samples_leaf = best_params['min_samples_leaf'],
        subsample = best_params['subsample'],
        random_state=42)
# Train the model

print(f'最优参数 : {best_params}')

print('使用最优参数训练模型...')
gbc.fit(X_train, y_train)
# Predict on the testing set
y_pred = gbc.predict(X_test)
# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率为: {accuracy}')
print(f'测试集样本: {y_test}')
print(f'模型预测: {y_pred}')
