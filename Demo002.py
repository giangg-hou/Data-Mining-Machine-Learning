import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier


sns.set_theme(style="ticks", palette="pastel")
df = pd.read_csv('Customertravel1.csv', delimiter=',')

# print("\n======================================\n")
# print(df.head())
# print("\n\n")
df = df.rename(columns={'Target': 'Churn'})

# print("\n======================================\n")
# print(df.info())


# print("\n\n")
# print(df.describe())

# print(df.columns)

# Split the numeric and categorical features
num_features = ['Age']
ordinal_features = ['AnnualIncomeClass', 'ServicesOpted']
cat_features = ['FrequentFlyer', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot', 'Churn']

sns.displot(data=df, x='Age', hue='Churn', kde=True )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1 = sns.violinplot(data=df, x='Churn', y='Age', ax=ax1)
ax2 = sns.boxplot(data=df, x='Churn', y='Age', ax=ax2)
plt.tight_layout()


print(df['Churn'].value_counts())


sns.countplot(data=df, x='Churn')

def plot_categorical(feature):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(x=feature,
                  hue='Churn',
                  data=df,
                  ax=ax1)
    ax1.set_ylabel('Count')
    ax1.legend(labels=['Retained', 'Churned'])
    sns.barplot(x=feature,
                y='Churn',
                data=df,
                ax=ax2)
    ax2.set_ylabel('Churn rate')
    plt.tight_layout()

for feature in (ordinal_features + cat_features[:-1]):
  plot_categorical(feature)

cat_data = pd.DataFrame()
for feature in cat_features[:-1]:
  temp = pd.get_dummies(df[feature], prefix=feature)
  cat_data = pd.concat([cat_data, temp], axis=1)

df['AnnualIncomeClass'] = df['AnnualIncomeClass'].map({'Low Income':0,
                             'Middle Income':1,
                             'High Income':2})

scaler = MinMaxScaler().set_output(transform='pandas')

df[num_features] = scaler.fit_transform(df[num_features])

X = pd.concat([cat_data, df[ordinal_features], df[num_features]], axis=1)

y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None, random_state=100)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

models.sort_values('F1 Score', ascending=False)

print(models)
print("\n")

models = []
models.append(('LR', LogisticRegression(random_state = 100)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state = 100)))
models.append(('RF', RandomForestClassifier(random_state = 100)))
models.append(('SVM', SVC(gamma='auto', random_state = 100)))
models.append(('XGB', GradientBoostingClassifier(random_state = 100)))
models.append(("LightGBM", LGBMClassifier(random_state = 100)))
models.append(("CatBoost", CatBoostClassifier(random_state = 100, verbose = False)))

results = []
names = []
for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        msg = "%s: (%f)" % (name, accuracy)
        print(msg)
        
model = RandomForestClassifier(random_state=100)
param_grid = {
    'n_estimators': [100],
    'criterion': ['entropy', 'gini'],
    'bootstrap': [True, False],
    'max_depth': [6],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [2, 3, 5],
    'min_samples_split': [2, 3, 5]
}

rf_clf = GridSearchCV(estimator=model,
                      param_grid=param_grid,
                      scoring='balanced_accuracy',
                      cv=5,
                      verbose=False,
                      n_jobs=-1)

best_rf_clf = rf_clf.fit(X_train, y_train)

print(best_rf_clf)

y_pred = best_rf_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=best_rf_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf_clf.classes_)

disp.plot()

print("\n=======================================================================\n")
print(classification_report(y_test, y_pred))
print("\n=======================================================================\n")

models2 = []
# models2.append(('DecisionTree', DecisionTreeClassifier(random_state = 42)))
# models2.append(('RF', RandomForestClassifier( random_state = 42)))
# models2.append(('XGB', GradientBoostingClassifier( random_state = 42)))
# models2.append(("LightGBM", LGBMClassifier( random_state = 42)))
# models2.append(("CatBoost", CatBoostClassifier(random_state = 42, verbose = False)))

models2.append(('RandomForest', RandomForestClassifier(random_state=42)))
models2.append(('DecisionTree', DecisionTreeClassifier(random_state = 42)))
models2.append(('GradientBoosting', GradientBoostingClassifier(random_state=42)))
models2.append(('XGBoost', XGBClassifier(random_state=42)))
models2.append(('LightGBM', LGBMClassifier(random_state=42)))
models2.append(('CatBoost', CatBoostClassifier(random_state=42, verbose=False)))

# for name, model in models2:
  #   clf = model.fit(X_train, y_train)
  #   y_pred = clf.predict(X_test)
    
  #  # Evaluate model
  #   accuracy = accuracy_score(y_test, y_pred)
  #   cm = confusion_matrix(y_test, y_pred)
  #   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    
  #   # Print evaluation metrics
  #   print(f"{name} Model Accuracy: {accuracy}")
  #   print(classification_report(y_test, y_pred))
    
  #   # Plot confusion matrix
  #   disp.plot()
  #   plt.title(f"{name} Confusion Matrix")
  #   plt.show()
    
    # base = model.fit(X_train,y_train)
    # y_pred = base.predict(X_test)
    # acc_score = accuracy_score(y_test, y_pred)
    # feature_imp = pd.Series(base.feature_importances_,
    #                 index=X.columns).sort_values(ascending=False)
    
    # sns.barplot(x=feature_imp, y=feature_imp.index)
    # plt.xlabel('Imoprtance score')
    # plt.ylabel('Features')
    # plt.title(name)
    # plt.show()