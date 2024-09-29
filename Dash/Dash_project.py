"""## Libraries"""
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


"""## Accessing data from GitHub"""

# Gitbhub's repository with the dataset.
url = "https://raw.githubusercontent.com/SantiagoMorenoV/Breast_Cancer_Logit_Model/refs/heads/main/breast-cancer-wisconsin.data"

headers = [
    "Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
    "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"
]

data = pd.read_csv(url, header=None, names=headers)
# Bare Nuclei contains some missing data with "?"
data.replace("?", pd.NA, inplace=True)
data["Bare Nuclei"] = pd.to_numeric(data["Bare Nuclei"]).astype('Int64')
dataset = data.dropna()

#dataset.info()

"""### Explanatory and explained variables"""

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"""## Splitting the dataset into Training and Test set"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

"""## GridSearch"""

model = LogisticRegression(random_state=0, max_iter=10000)
param_grid = {
    'C': [1, 5, 10, 50, 100],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

"""## Logistic Classifier"""

classifier = LogisticRegression(C = 1,random_state = 0, solver = 'lbfgs')
classifier.fit(X_train, y_train)

"""### Predicting"""

y_pred = classifier.predict(X_test)

"""### Confussion Matrix"""

cm = confusion_matrix(y_test, y_pred)

"""### ROC Curve"""

"""# Converting y_train and y_test values to {0, 1}"""
y_train_bin = (y_train == 4).astype(int)
y_test_bin = (y_test == 4).astype(int)

# Adjusting the model with the binirized data 
classifier.fit(X_train, y_train_bin)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

# Calculating the true and false positive rates tor the ROC curve
fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba)

# Calculating the area under the curve (AUC) with more decimals
roc_auc = np.round(auc(fpr, tpr), decimals=6) * 100

"""## Dash App"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Results summary: Logit (Breast Cancer Tumor Classification)", className="text-center"), className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='confusion-matrix'), width=6),
            dbc.Col(dcc.Graph(id='roc-curve'), width=6)
        ]),
        dbc.Row([
            dbc.Col(html.Div([
                html.H3("GridSearch Best Hyperparameters"),
                html.P(f"C: {best_params['C']}"),
                html.P(f"Solver: {best_params['solver']}")
            ]), width=12)
        ])
    ])
])

"""### App callback"""

@app.callback(
    [Output('confusion-matrix', 'figure'),
     Output('roc-curve', 'figure')],
    [Input('confusion-matrix', 'id')]
)
def update_graphs(_):
    # Matriz de confusión
    z = cm
    x = ['Predicted 0', 'Predicted 1']
    y = ['Actual 0', 'Actual 1']
    z_text = [[str(y) for y in x] for x in z]

    fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Blues')
    fig_cm.update_layout(title_text='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')

    # Curva ROC
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

    return fig_cm, fig_roc

if __name__ == '__main__':
    #app.run(debug=True, jupyter_mode="external") # It displays the Dashboard runing on la local site
    app.run(debug=True)