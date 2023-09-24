from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import GridSearchCV


df = pd.DataFrame({
    'Fare': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5], 
    'Embarked': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    'Name': ['Ash', 'Brock', 'Misty', 'Oak', 'Meme', 'Ash', 'Brock', 'Misty', 'Oak', 'Meme', 'Ash', 'Brock', 'Misty', 'Oak', 'Meme', 'Ash', 'Brock', 'Misty', 'Oak', 'Meme'],
    'Sex': ['male', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female'],
    'Age': [10, 20, 23, 40, 30, 10, 20, 23, 40, 30, 10, 20, 23, 40, 30, 10, 20, 23, 40, 30],
    })

X = df.drop(columns=['Embarked'])
y = df['Embarked']

ohe = OneHotEncoder()
vect = CountVectorizer()
ct = make_column_transformer((ohe, ['Sex']), (vect, 'Name'))
clf = LogisticRegression(solver='liblinear', random_state=1)


# pipe = make_pipeline(ct, clf)

pipe = Pipeline(steps=[('ct', ct), ('clf', clf)])

cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

# params = {}
# params['columntransformer__countvectorizer__min_df'] = [1, 2]
# params['logisticregression__C'] = [0.1, 1, 10]
# params['logisticregression__penalty'] = ['l1', 'l2']

params = {}
params['ct__countvectorizer__min_df'] = [1, 2]
params['clf__C'] = [0.1, 1, 10]
params['clf__penalty'] = ['l1', 'l2']


grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')

grid.fit(X, y)

print(grid.best_score_)
