import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['target'] = pd.Categorical.from_codes(iris['target'], iris['target_names'])