import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

plt.style.use('ggplot')

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
data = pd.read_csv(url, header=None, na_values='?')

data.columns = ['A'+ str(i) for i in range(1,16)]+['class']
category_object = [c for c in data.columns if data[c].dtype.name == 'object']
category_numeric = [d for d in data.columns if data[d].dtype.name != 'object' ]
c = data.describe(include=[object])

# for c in category_object:
#     print(data[c].unique())

from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.05, figsize=(10,10))
# plt.show()


col1 = 'A2'
col2 = 'A14'

plt.figure(figsize=(10, 6))
plt.scatter(data[col1][data['class'] == '+'],
            data[col2][data['class'] == '+'],
            alpha = 0.75,
            color='red',
            label = '+')

plt.scatter(data[col1][data['class'] =='-'],
            data[col2][data['class'] == '-'],
            alpha=0.75,
            color = 'blue',
            label = '-')
plt.xlabel(col1)
plt.xlabel(col2)
plt.legend(loc='best')
# plt.show()
# print(data.corr())

print(data['A1'].describe())

dataFill = data.fillna(data.median(axis=0), axis=0)
print(dataFill)

data_describe = data.describe(include=[object])
for c in category_object:
    data[c] = data[c].fillna(data_describe[c]['top'])
print(data[c])
data.describe(include=[object])


# missing data
# print(data.count(axis=0))
# data1 = data.dropna(axis=1)
# data2 = data.dropna(axis=0)
# print(data1)
# print(data2)


# print(category_numeric)
# print(category_object)
# print(data[category_object].describe())
# print(c)

# print(data.shape)
# print(data.head())
# print(data.tail())
# print(data['A8'][452])
# print(data.describe())


binary_columns    = [c for c in category_object if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in category_object if data_describe[c]['unique'] > 2]
print(binary_columns, nonbinary_columns)

data.at[data['A1'] == 'b', 'A1']=0
data.at[data['A1'] == 'a', 'A1']=1

data_describe = data.describe(include=[object])
print(data_describe)

for c in binary_columns [1:]:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1
a = data[binary_columns].describe()
print(a)