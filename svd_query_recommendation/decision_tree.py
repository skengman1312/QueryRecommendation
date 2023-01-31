import numpy as np
import pandas as pd
from typing import Optional, List
from numpy.linalg import svd
from svd_query_recommendation import *

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text


class DT:
    def __init__(self, um):
        self.um = um
        np.random.seed(1312)
        if um.filled_matrix is None:
            um.fill()
        self.u, self.s, self.vh = svd(um.filled_matrix, full_matrices=False)

        # keep as many concepts as we need to explain 90% of the variance
        explained_variance = np.array([s_i**2 / sum(self.s**2) for s_i in self.s])
        for idx in range(len(explained_variance))[::-1]:
            if sum(explained_variance[:idx]) < 0.9:
                self.top_n_concepts = idx + 1
                break

    def get_top_q(self,
                  top_n_queries: Optional[int] = 10
                 ) -> List[pd.DataFrame]:
        self.top_q = {idx: item.argpartition(-top_n_queries)[-top_n_queries:] for idx, item in enumerate(self.vh[:self.top_n_concepts])}
        # using the indexes grab the data points which satisfy the corresponding queries
        self.top_res = pd.concat([pd.concat([self.um.dataset.query(self.um.queries[q]) for q in value]).assign(concept=key)
                                  for key, value in self.top_q.items()])
        return self.top_res
    

if __name__ == "__main__":
    um = UtilityMatrix.from_dir("../discreate_small/")
    decision_tree = DT(um)
    for q in [3]:
        print(f'\nq: {q}')

        dataset = decision_tree.get_top_q(top_n_queries=q)
        print(f'original data shape:                  {dataset.shape}')
        dataset = dataset.drop_duplicates(subset=list(dataset.columns[:-1]), keep='first')
        print(f'data shape after dropping duplicates: {dataset.shape}')

        X = dataset.iloc[:, :-1]
        X = pd.concat([pd.get_dummies(X[col], prefix=col) for col in X.columns], axis=1)
        y = dataset.iloc[:, -1]
        print()
        print(round(y.value_counts(normalize=True)*100, 2))

        clf = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=decision_tree.top_n_concepts, max_depth=int(decision_tree.top_n_concepts/2))  # decision_tree.top_n_concepts)
        print()
        print(f'cross_val accuracy: {cross_val_score(clf, X, y, cv=3, scoring="accuracy")}')
        clf.fit(X, y)

        print(export_text(clf))

        temp = pd.concat([pd.get_dummies(um.dataset.table[col], prefix=col) for col in um.dataset.table.columns], axis=1)
        temp = temp.iloc[:5, :]
        temp['concept'] = clf.predict(temp)
        print(temp.head())

        n_nodes = clf.tree_.node_count
        node_indicator = clf.decision_path(temp.iloc[:, :-1])
        print(node_indicator)

        for concept_num in range(9):
            print(f'\nconcept_num: {concept_num}')
            sample_ids = temp.index[temp.concept == concept_num].tolist()
            # boolean array indicating the nodes both samples go through
            common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
            # obtain node ids using position in array
            common_node_id = np.arange(n_nodes)[common_nodes]

            print(
                "\nThe following samples {samples} share the node(s) {nodes} in the tree.".format(
                    samples=sample_ids, nodes=common_node_id
                )
            )
            print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))

"""
- check if all users are associated to a concept or not
- get some negative examples for training of the model with a special label or smthing
- based on the shared deciding nodes, reconstruct the query from that
- 
"""
