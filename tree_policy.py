"""Array-only inference for compact exported action-value trees."""
import numpy as np


class ArrayTrees:
    def __init__(self, data):
        self.feature = data['tree_feature']
        self.left = data['tree_left']
        self.right = data['tree_right']
        self.threshold = data['tree_threshold']
        self.value = data['tree_value']
        self.base = data['tree_base']
        self.depth = int(data['tree_depth'])
        self.rows, self.columns = np.indices(self.feature.shape[:2])

    def predict(self, features):
        node = np.zeros(self.feature.shape[:2], dtype=np.int32)
        for _ in range(self.depth):
            index = self.rows, self.columns, node
            go_left = features[self.feature[index]] <= self.threshold[index]
            node = np.where(go_left, self.left[index], self.right[index])
        return self.base + self.value[self.rows, self.columns, node].sum(axis=1)
