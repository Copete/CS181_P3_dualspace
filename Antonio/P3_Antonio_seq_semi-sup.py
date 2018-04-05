## P3: Classifying Sounds
## Antonio Copete



import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import RepeatedStratifiedKFold

def sequential_clf(X_train, y_train, X_test, semi_sup=False, #True for semi-sup
                   classifier=LabelSpreading,
                   **kwargs): # Additional keywords for classifier
    '''
    Classify sequentially by class (most accurate & smallest first)
    Option to use semi-supervised learning (LabelSpreading classifier)
    '''
    classes = np.array(['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'])
    n_splits = 5  # No. of K-folds for cross-validation
    n_repeats = 3 # No. of cross-validation repeats
    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]
    assert(len(y_train) == n_train)
    n_classes = len(classes)
    iclasses, class_cts = np.unique(y_train, return_counts=True)
    assert(np.sum(np.equal(iclasses, np.arange(n_classes))) == n_classes)
    Y_train = np.zeros((n_train, n_classes), dtype=bool)
    Y_train[np.arange(n_train), y_train] = True
    y_test_pred = np.full(n_test, -1)
    ih = np.arange(n_test)
    for i in range(n_classes-1):
        print('{}) {} classes:'.format(i+1, len(iclasses)))
        # Fit classifier on current classes
        clf = classifier(**kwargs)
        Y_train_ = Y_train[:,iclasses]
        ix = (np.sum(Y_train_,axis=1) > 0).nonzero()[0]
        # Score classes
        score_cl = []
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=42)
        for i_train0, i_val0 in rskf.split(X_train[ix], Y_train_[ix].argmax(axis=1)):
            i_train = ix[i_train0]
            i_val   = ix[i_val0]
            X_train_, y_train_ = X_train[i_train], Y_train_[i_train].argmax(axis=1)
            X_val_,   y_val_   = X_train[i_val],   Y_train_[i_val].argmax(axis=1)
            n_val_ = len(i_val0)
            # Semi-supervised vs. Supervised fitting
            if semi_sup:
                clf.fit(np.vstack((X_train_,X_val_)),
                        np.concatenate((y_train_,np.full(n_val_,-1))))
            else:
                clf.fit(X_train_, y_train_)
            score_cl0 = [clf.score(X_val_[y_val_==icl],
                                   y_val_[y_val_==icl]) for icl in range(len(iclasses))]
            score_cl0 = np.append(score_cl0, clf.score(X_val_, y_val_))
            score_cl.append(score_cl0)
        score_cl = np.array(score_cl)
        score_cl_std = 100*np.std(score_cl,axis=0)/np.mean(score_cl,axis=0)
        score_cl = np.mean(score_cl, axis=0)
        for icl in range(len(iclasses)):
            print('   Class {:16s} ({:4d} cts): {:.4f} ± {:.2f}%'.format(classes[icl], class_cts[icl], score_cl[icl], score_cl_std[icl]))
        print('   Overall CV score ({}x {}-fold): {:.4f} ± {:.2f}%'.format(n_repeats, n_splits, score_cl[-1], score_cl_std[-1]))
        # Select best class
        icl = np.argsort(class_cts)
        imax = icl[np.argmax(score_cl[icl])]
        score_imax = score_cl[imax]
        # Repeat cross-validation for one-vs-rest
        y_train0 = np.zeros(len(ix), dtype=int)
        y_train0[y_train[ix] == iclasses[imax]] = 1
        score_imax_ovr = []
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=42)
        for i_train, i_val in rskf.split(X_train[ix], y_train0):
            X_train_, y_train_ = X_train[ix[i_train]], y_train0[i_train]
            X_val_,   y_val_   = X_train[ix[i_val]],   y_train0[i_val]
            n_val_ = len(i_val)
            if semi_sup:
                clf.fit(np.vstack((X_train_,X_val_)),
                        np.concatenate((y_train_,np.full(n_val_,-1))))
            else:
                clf.fit(X_train_, y_train_)
            score_imax_ovr.append(clf.score(X_val_[y_val_==1], y_val_[y_val_==1]))
        score_imax_ovr = np.mean(score_imax_ovr)
        # Make predictions for best class in one-vs-rest fashion
        X_train_ = np.vstack((X_train[ix], X_test[ih]))
        y_train_ = np.zeros(len(ix), dtype=int)
        y_train_[y_train[ix] == iclasses[imax]] = 1
        y_train_ = np.append(y_train_, np.full(len(ih),-1))
        clf.fit(X_train_, y_train_)
        y_test_pred_ = clf.predict(X_test[ih])
        y_test_pred[ih[y_test_pred_==1]] = iclasses[imax]
        ih = ih[y_test_pred_ != 1]
        print('     Best class: {:10s} ({:3d} files, {:.3f}% of training data)'.format(classes[imax],
                class_cts[imax], 100*class_cts[imax]/n_train))
        print('        CV Score (3x 5-fold): {:.4f} ({:.4f} one-vs-rest)'.format(score_imax, score_imax_ovr))
        print('        Classified test files: {}, {:.3f}% of test data'.format(np.sum(y_test_pred_==1),
                100*np.sum(y_test_pred_==1)/n_test))
        classes = np.delete(classes, imax)
        iclasses = np.delete(iclasses, imax)
        class_cts = np.delete(class_cts, imax)
        if i == n_classes-2:
            y_test_pred[ih] = iclasses[0]
            print('     Remaining class: {:10s} ({:3d} files, {:.3f}% of training data)'.format(classes[0],
                class_cts[0], 100*class_cts[0]/n_train))
            print('        Classified test files: {}, {:.3f}% of test data'.format(len(ih),
                100*len(ih)/n_test))
    return y_test_pred

# Example using semi-supervised classifier
y_test_pred = sequential_clf(X_train, # Training set
                y_train,  # Class labels (1-D form!!!)
                X_test,   # Test set
                semi_sup=True, # Use semi-sup classification
                classifier=LabelSpreading, # Any other classifier works
                #LabelSpreading params
                kernel='knn',  # Can also be 'rbf' for SVM radial kernel
                gamma=2,       # Only for kernel='rbf'
                n_neighbors=5, # Only for kernel='knn'
                alpha=0.0001) # Clamping factor (0-1, 0=don't change init. labels)

# Can feed predictions directly into routine to write results
#write_to_file("P3_pred.csv", X_test_ids, y_test_pred)
