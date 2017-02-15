import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    print "Loading data..."
    df = pd.read_csv('../input/train.csv')
    eval_df = pd.read_csv('../input/test.csv')

    target = 'label'

    print "Splitting data into train/test sets..."
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2, random_state=1)

    pca = PCA(n_components=3)
    pca_transform = pca.fit_transform(X_train)
    var_values = pca.explained_variance_ratio_

    print "Setting up GridSearchCV..."
    svc_param = [{'kernel': ['linear', 'rbf', 'sigmoid']
                     , 'gamma': [1e-3, 1e-4, 1e-5]
                     , 'C': np.arange(1000, 10000, 1000)}]

    nn_param = [{'alpha': [1e-2, 1e-3, 1e-4]
                    , 'momentum': np.arange(0.3, 0.6, 0.05)
                    , 'hidden_layer_sizes': [(10,5), (20,10), (40,20), (60,30), (80,40), (100,50)]
                }]

    clf = GridSearchCV(estimator=SVC(cache_size=8000, decision_function_shape='ovr', random_state=1), n_jobs=3, cv=5, param_grid=svc_param)
    print "Fitting model..."
    clf.fit(X_train, y_train)

    print "CV Results: "
    print clf.cv_results_
    print "Best estimator found by GridSearchCV: "
    print clf.best_estimator_
    print "with a score of: "
    print clf.best_score_

    print "Predicting on test set..."
    y_pred = clf.predict(X_test)

    print("Classification report for model %s:\n%s\n"
          % (clf, classification_report(y_test, y_pred)))
    print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred))

    print "Creating kaggle submission file..."
    predictions = clf.predict(eval_df.drop(target, axis=1))
    submission = pd.DataFrame({"id": eval_df['id'], 'type': predictions})
    submission.to_csv("output/digitRecognizerSubmission.csv", index=False)