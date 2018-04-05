import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Needs X_train, Y_train to be provided

# Evaluate current feature set by 5-fold CV
rf = RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                            n_estimators=1000, max_features='sqrt')
cv_scores = cross_val_score(rf, X_train, Y_train, cv=5)
print('CV scores: ', cv_scores, 'Mean: {:.4f}'.format(np.mean(cv_scores)))
rf.fit(X_train2_, Y_train2_)
print('Holdout set score: ', rf.score(X_holdout, Y_holdout))

# Fit entire training dataset and plot feature importances
rf.fit(X_train, Y_train)
n_feat = np.array([X_train_mfcc.shape[1], X_train_mfcc.shape[1],
                   X_train_mfcc_delta.shape[1], X_train_mfcc_delta2.shape[1],
                   X_train_chroma.shape[1], X_train_mel.shape[1], #[:,:40].shape[1],
                   X_train_contrast.shape[1], X_train_tonnetz.shape[1]])
feat_name = np.array(['MFCC', r'MFCC-$\sigma$', r'MFCC-$\Delta$', r'MFCC-$\Delta^2$',
                      'Chroma', 'MEL', 'Contrast', 'Tonnetz'])
imp = rf.feature_importances_
sns.set()
plt.clf()
plt.figure(figsize=(5.5,2.5))
i0 = 0
for ifeat,n_feat0 in enumerate(n_feat):
    plt.bar(i0+np.arange(n_feat0), imp[i0:i0+n_feat0], label=feat_name[ifeat]+' ({})'.format(n_feat0))
    i0 += n_feat0
plt.title('Initial set of features')
plt.xlabel('Features')
plt.ylabel('Feature importances')
plt.legend(title='Feature Categories', loc=(1,0))
plt.savefig('P3_fig_feature_imp.pdf',bbox_inches='tight')
plt.show()
