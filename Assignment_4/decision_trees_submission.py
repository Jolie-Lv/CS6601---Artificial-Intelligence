from __future__ import division

import numpy as np
from collections import Counter
import time



class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = map(int, out[:, class_index])
        features = out[:, :class_index]
        return features, classes

    elif class_index == 0:
        classes = map(int, out[:, class_index])
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.

    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = None
    decision_tree_root = DecisionNode(None,None,lambda feature:feature[0]==1)
    decision_tree_root.left = DecisionNode(None,None,None,1)
    decision_tree_root.right = DecisionNode(None,None,lambda feature:feature[3]==1)
    decision_tree_root.right.left = DecisionNode(None,None,lambda feature:feature[1]==0)
    decision_tree_root.right.right = DecisionNode(None,None,lambda feature:feature[2]==1)
    decision_tree_root.right.left.left = DecisionNode(None,None,None,1)
    decision_tree_root.right.left.right = DecisionNode(None,None,None,0)
    decision_tree_root.right.right.left = DecisionNode(None,None,None,0)
    decision_tree_root.right.right.right = DecisionNode(None,None,None,1)
    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    true_pos = 0.0
    true_neg = 0.0
    false_neg = 0.0
    false_pos = 0.0
    for elem1,elem2 in zip(classifier_output, true_labels):
        if(elem1==elem2) and (elem1==1):
            true_pos += 1
        elif(elem1==elem2) and (elem2!=1):
            true_neg += 1
        elif(elem1 != 1):
            false_neg +=1
        else:
            false_pos +=1
    conf_matrix = np.array([[true_pos, false_neg],[false_pos, true_neg]])
    return conf_matrix


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    conf_matrix = confusion_matrix(classifier_output, true_labels)
    return conf_matrix[0][0]/(conf_matrix[0][0] + conf_matrix[1][0])


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    conf_matrix = confusion_matrix(classifier_output, true_labels)
    return conf_matrix[0][0]/(conf_matrix[0][0] + conf_matrix[0][1])


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    conf_matrix = confusion_matrix(classifier_output, true_labels)
    return (conf_matrix[0][0]+conf_matrix[1][1])/(conf_matrix[0][0] + conf_matrix[0][1]\
           + conf_matrix[1][0] + conf_matrix[1][1])


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """
    total = len(class_vector)
    num_pos = class_vector.count(1)
    num_neg = total - num_pos
    if total > 0:
        prob_pos = num_pos/float(total)
        prob_neg = num_neg/float(total)
        return 2*prob_pos*prob_neg
    else:
        return -1


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    I_parent = gini_impurity(previous_classes)
    I_child = 0
    for elem in current_classes:
        I_child += len(elem)/float(len(previous_classes))*gini_impurity(elem)
    return I_parent - I_child


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float(20)):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)
        
    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        root = None
        if (len(set(classes)) <= 1) and (len(classes) != 0) :
            return DecisionNode(None,None,None,classes[0])
        elif (len(classes) == 0):
            return DecisionNode(None,None,None,2)
        elif depth == self.depth_limit:
            return DecisionNode(None,None,None,max(set(classes), key=list(classes).count))
        else:
#            if depth == 0:
            features = np.array(features)
            classes = np.array(classes).reshape(-1,1)
            feat_shape = features.shape
            sample_list = range(feat_shape[0])
            gains = np.zeros((feat_shape[1]))
            indices = np.zeros((feat_shape[1]))
            for i in range(feat_shape[1]):
                attribute =  features[:,i]
                for j in range(20):
                    split_indx = int(np.random.choice(sample_list, replace=False))
                    idx_above = np.where(attribute > attribute[split_indx])[0]
                    idx_below = np.where(attribute < attribute[split_indx])[0]
                    classes_below = classes[idx_below,:].reshape(1,-1)[0]
                    classes_above = classes[idx_above,:].reshape(1,-1)[0]
                    gain = gini_gain(list(classes.reshape(1,-1)[0]),[list(classes_below),list(classes_above)])
                    if gain > gains[i]:
                        gains[i] = gain
                        indices[i] = split_indx
            indx = np.argmax(gains)
            split_indx = int(indices[indx])
            attribute =  features[:,indx]
            idx_above = np.where(attribute > attribute[split_indx])[0]
            idx_below = np.where(attribute < attribute[split_indx])[0] 
            features_below = features[idx_below,:]
            features_above = features[idx_above,:]
            classes_below = classes[idx_below,:].reshape(1,-1)[0]
            classes_above = classes[idx_above,:].reshape(1,-1)[0]
            if (len(classes_below) != 0) and (len(classes_above) != 0):
                root = DecisionNode(None,None,lambda feat:feat[indx] > features[split_indx,indx])
                root.left = self.__build_tree__(features_above, classes_above, depth+1)
                root.right = self.__build_tree__(features_below, classes_below, depth+1)
                return root
            elif (len(classes_below) == 0) and (len(classes_above) != 0):
                return DecisionNode(None,None,None,max(set(classes_above), key=list(classes_above).count))
            elif (len(classes_above) == 0) and (len(classes_below) !=0):
                return DecisionNode(None,None,None,max(set(classes_below), key=list(classes_below).count))
            else:
                return DecisionNode(None,None,None,2)


                


    def classify(self, features):
        """Use the fitted tree to classify a list of example features.

        Args:
            features (list(list(int)): List of features.

        Return:
            A list of class labels.
        """
        
        class_labels = []
        # TODO: finish this.
        features = np.array(features)
        feat_shape = features.shape
        for indx in range(feat_shape[0]):
#            print list(features[indx,:]), features[indx,:]
            decision = self.root.decide(list(features[indx,:]))
            class_labels.append(decision)
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """

    # TODO: finish this.
    folds = []
    dataset = np.concatenate((dataset[0], np.array(dataset[1]).reshape(-1,1)), axis=1)
    dataset_shape = dataset.shape
    shape_test_set = int(round(dataset_shape[0]/k,0))
    split_dataset = np.array_split(dataset,k,axis=0)
    for i in range(k):
        test_set = split_dataset[i]
        c = [k for j,k in enumerate(split_dataset) if j!=i]
        training_set = np.concatenate(c,axis=0)
        if test_set.shape[0] != shape_test_set:
            step = test_set.shape[0] - shape_test_set
            test_set = test_set[:-step,:]
            training_set = np.concatenate((training_set, test_set[-step:,:]), axis=0)
        r_test_set = (test_set[:,:-1], list(test_set[:,-1]))
        r_train_set = (training_set[:,:-1], list(training_set[:,-1]))
        folds.append((r_train_set, r_test_set))
    return folds
        
        


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.attr_track = []

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        features = np.array(features)
        classes = np.array(classes).reshape(-1,1)
        #print classes.shape
        feat_shape = features.shape
        num_sample = int(self.example_subsample_rate*feat_shape[0])
        num_attr = int(self.attr_subsample_rate*feat_shape[1])
        #print num_attr, self.attr_subsample_rate
        for i in range(self.num_trees):
            idx = np.random.randint(feat_shape[0],size=num_sample)
            sampled_features = features[idx,:]
            sampled_classes = classes[idx,:].reshape(1,-1)[0]
            sampled_attr = np.random.choice(range(feat_shape[1]),num_attr,replace=False)
            #print sampled_attr, feat_shape[1], num_attr
            self.attr_track.append(sampled_attr)
            tree = DecisionTree(depth_limit=self.depth_limit)
            tree.fit(sampled_features[:,sampled_attr],sampled_classes)
            self.trees.append(tree)
            
        

    def classify(self, features):
        """Classify a list of features based on the trained random forest.

        Args:
            features (list(list(int)): List of features.
        """

        # TODO: finish this.
        class_labels = []
        # TODO: finish this.
        features = np.array(features)
        feat_shape = features.shape
        for i in range(feat_shape[0]):
            vote = np.zeros((self.num_trees))
            for j in range(self.num_trees):
                #print self.trees[j].classify(feat)
                vote[j] = self.trees[j].classify(features[i,self.attr_track[j]].reshape(1,-1))[0]
            counts = np.bincount(vote.astype(int))
            class_labels.append(np.argmax(counts))
        return class_labels


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self,num_trees=100, depth_limit=5, example_subsample_rate=0.4,
                 attr_subsample_rate=0.4):
        """Create challenge classifier.

        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.classifier = RandomForest(self.num_trees, self.depth_limit, self.example_subsample_rate,
                 self.attr_subsample_rate)

    def fit(self, features, classes):
        """Build the underlying tree(s).

            Fit your model to the provided features.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        classes = np.array(classes)
        features = np.array(features)
        idx_1 = np.where(classes == 1)[0]
        idx_0 = np.where(classes == 0)[0]
        new_features = np.concatenate((features[idx_0,:], features[idx_1,:]), axis=0)
        new_classes = np.concatenate((classes[idx_0], classes[idx_1]), axis=0)
        
        self.classifier.fit(new_features, new_classes)

    def classify(self, features):
        """Classify a list of features.

        Classify each feature in features as either 0 or 1.

        Args:
            features (list(list(int)): List of features.

        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        features = np.array(features)
        return self.classifier.classify(features)


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Args:
            data: data to be added to array.

        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Bonnie time to beat: 0.09 seconds.

        Args:
            data: data to be sliced and summed.

        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        return np.add(np.multiply(data,data), data)

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Args:
            data: data to be added to array.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Bonnie time to beat: 0.07 seconds

        Args:
            data: data to be sliced and summed.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        sum_of_rows = np.sum(data[:100,:], axis=1)
        max_indx = np.argmax(sum_of_rows)
        return (sum_of_rows[max_indx], max_indx)

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Bonnie time to beat: 15 seconds

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        flatten_data = data.flatten()
        flatten_data = flatten_data[flatten_data>0]
        return Counter(flatten_data).items()
        
def return_your_name():
    # return your name
    # TODO: finish this
    return "Dan Monga Kilanga"
