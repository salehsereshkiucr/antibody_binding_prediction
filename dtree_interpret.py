import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb


def interpret(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("The binary tree structure has {n} nodes and has the following tree structure:\n".format(n=n_nodes))
    #for i in range(n_nodes[:10]):
    for i in range(10):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: ""go to node {left} if X[:, {feature}] <= {threshold} else to node {right}.".format(
                    space=node_depth[i] * "\t",node=i,left=children_left[i],feature=feature[i],threshold=threshold[i],right=children_right[i],))
    return is_leaves, feature, threshold, children_left, children_right, node_depth


is_leaves, feature, threshold, children_left, children_right, node_depth = interpret(clf)


feature_weights = {}
for i in range(np.max(feature)+1):
    feature_weights[i] = 0
n_weight = (np.max(node_depth) + 1) - node_depth
for i in range(len(feature)):
    if feature[i] >= 0:
        feature_weights[feature[i]] = feature_weights[feature[i]] + n_weight[i]
res = np.zeros(36*22)
for k in feature_weights:
    res[k] = feature_weights[k]

res = res.reshape((36, 22)).T
res = (res-np.min(res))/(np.max(res)-np.min(res))

plt.imshow(res, cmap='Greys')
plt.savefig('feature_weights_PD.png')
plt.close()

res = clf.feature_importances_.reshape((36, 21))
res = (res-np.min(res))/(np.max(res)-np.min(res))
feature_importances = pd.DataFrame(res)
feature_importances[22] = feature_importances[feature_importances.columns].sum(axis=1)
row_names = encode_mat.columns

fig, ax = plt.subplots(figsize=(11, 9))
sb.heatmap(feature_importances, cmap="Blues", linewidth=0.3)
ax.xaxis.tick_bottom()
xticks_labels = list(encode_mat.columns) + ['sum']
plt.xticks(np.arange(22) + .5, labels=xticks_labels)
plt.yticks([i*5 + 0.5 for i in range(8)], labels=[i*5 for i in range(8)])
plt.xlabel('Amino Acid')
plt.ylabel('Position')
plt.title('PD-1', loc='center', fontsize=16)
plt.savefig('PD-1_feature_importance.png')
plt.close()
