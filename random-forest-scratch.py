import pandas as pd
import numpy as np

def entropy_multi(y):
    if len(y) == 0:
        return 0
    y = np.array(y).astype(int)
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])


def information_gain(left_child, right_child):
    parent = np.concatenate([left_child, right_child])
    IG_p = entropy_multi(parent)
    IG_l = entropy_multi(left_child)
    IG_r = entropy_multi(right_child)
    return IG_p - (len(left_child)/len(parent))*IG_l - (len(right_child)/len(parent))*IG_r


def terminal_node(node):
    y_bootstrap = node['y_bootstrap']
    if len(y_bootstrap) == 0:
        return None
    counts = np.bincount(y_bootstrap)
    pred = np.argmax(counts)
    return pred


def split_node(node, max_features, min_samples_split, max_depth, depth):
    left_child = node['left_child']
    right_child = node['right_child']

    del(node['left_child'])
    del(node['right_child'])

    if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
        empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
        node['left_split'] = terminal_node(empty_child)
        node['right_split'] = terminal_node(empty_child)
        return

    if depth >= max_depth:
        node['left_split'] = terminal_node(left_child)
        node['right_split'] = terminal_node(right_child)
        return node
    
    if node['information_gain'] is not None and node['information_gain'] < 1e-5:
        node['left_split'] = terminal_node(left_child)
        node['right_split'] = terminal_node(right_child)
        return node

    if len(left_child['X_bootstrap']) <= min_samples_split:
        node['left_split'] = node['right_split'] = terminal_node(left_child)
    else:
        node['left_split'] = find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
        split_node(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
    if len(right_child['X_bootstrap']) <= min_samples_split:
        node['right_split'] = node['left_split'] = terminal_node(right_child)
    else:
        node['right_split'] = find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
        split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)

def find_split_point(X_bootstrap, y_bootstrap, max_features):
    num_features = X_bootstrap.shape[1]
    max_features = min(max_features, num_features)  # segurança

    features_list = np.random.choice(range(num_features), size=max_features, replace=False)

    best_information_gain = -np.inf
    node = None
    for feature_index in features_list:
        feature_values = X_bootstrap[:, feature_index]
        split_points = np.unique(feature_values)
        
        for split_point in split_points:
            left_mask = feature_values <= split_point if isinstance(split_point, (int, float)) else feature_values == split_point
            
            left_child = {
                'X_bootstrap': X_bootstrap[left_mask],
                'y_bootstrap': y_bootstrap[left_mask]
            }
            right_child = {
                'X_bootstrap': X_bootstrap[~left_mask],
                'y_bootstrap': y_bootstrap[~left_mask]
            }

            if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
                continue

            split_info_gain = information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
            if split_info_gain > best_information_gain:
                best_information_gain = split_info_gain

                left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
                right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
                node = {'information_gain': split_info_gain,
                        'left_child': left_child,
                        'right_child': right_child,
                        'split_point': split_point,
                        'feature_idx': feature_index}
                
    return node



def draw_bootstrap(X_train, y_train):
    bootstrap_i = list(np.random.choice(range(len(X_train)), len(X_train), replace=True))
    oob_i = [i for i in range(len(X_train)) if i not in bootstrap_i]

    X_bootstrap = X_train.iloc[bootstrap_i].values
    y_bootstrap = y_train[bootstrap_i]
    
    X_oob = X_train.iloc[oob_i].values
    y_oob = y_train[oob_i]
    
    return X_bootstrap, y_bootstrap, X_oob, y_oob

def oob_score(tree, X_test, y_test):
    mis_label = 0
    for i in range(len(X_test)):
        pred = predict_tree(tree, X_test[i])
        if pred != y_test[i]:
            mis_label += 1
    return mis_label / len(X_test)

def split_data(data, features, perc):
    nb_train = int(np.floor(perc * len(data)))
    data = data.sample(frac=1, random_state=217)

    X_train = data[features][:nb_train]
    y_train = data['Class'][:nb_train].values
    X_test = data[features][nb_train:]
    y_test = data['Class'][nb_train:].values

    return X_train, y_train, X_test, y_test

def build_tree(X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
    root_node = find_split_point(X_bootstrap, y_bootstrap, max_features)
    if root_node is not None:
        split_node(root_node, max_features, min_samples_split, max_depth, 1)
    else:
        root_node = {'left_split': terminal_node({'y_bootstrap': y_bootstrap}),
                     'right_split': terminal_node({'y_bootstrap': y_bootstrap}),
                     'feature_idx': 0,
                     'split_point': 0}
    return root_node

def random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
    tree_ls = list()
    oob_ls = list()
    for i in range(n_estimators):
        X_bootstrap, y_bootstrap, X_oob, y_oob = draw_bootstrap(X_train, y_train)
        tree = build_tree(X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split)
        tree_ls.append(tree)
        oob_error = oob_score(tree, X_oob, y_oob)
        oob_ls.append(oob_error)
    print("OOB estimate: {:.2f}".format(np.mean(oob_ls)))
    return tree_ls

def predict_tree(tree, X_test):
    feature_idx = tree['feature_idx']

    if X_test[feature_idx] <= tree['split_point']:
        if type(tree['left_split']) == dict:
            return predict_tree(tree['left_split'], X_test)
        else:
            value = tree['left_split']
            return value
    else:
        if type(tree['right_split']) == dict:
            return predict_tree(tree['right_split'], X_test)
        else:
            return tree['right_split']
        
def predict_rf(tree_ls, X_test):
    pred_ls = list()
    for i in range(len(X_test)):
        ensemble_preds = [predict_tree(tree, X_test.values[i]) for tree in tree_ls]
        final_pred = max(ensemble_preds, key = ensemble_preds.count)
        pred_ls.append(final_pred)
    return np.array(pred_ls)

if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    features = ['P_sist', 'P_dist', 'qPA', 'Pulse', 'BreathFreq']
    data['Class'] = data['Class'].astype('category').cat.codes


    X_train, y_train, X_test, y_test = split_data(data, features, 0.5)

    # parameters
    n_estimators = 100
    max_features = 3
    max_depth = 5
    min_samples_split = 2

    # training
    tree_ls = random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split)

    # predict
    y_pred = predict_rf(tree_ls, X_test)

    # evaluation
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print("Acurácia: {:.2f}".format(accuracy))



