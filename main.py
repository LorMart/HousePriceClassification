# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:02:54 2024

@author: Loren
"""

# -*- coding: utf-8 -*-
""" 
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np
from sklearn.svm import SVC
from scipy.stats import chi2_contingency
from scipy.stats.contingency import crosstab

df = pd.read_csv('/train_house_price_range.csv')
test_df = pd.read_csv('/test_house_price_range.csv')


def replace_qualitative_values(df, columns, mapping):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(mapping)
    return df

def replace_nan_values(df, column_names):
    quantitative_columns = []
    qualitative_columns = []

    for column_name in column_names:
        column = df[column_name]
        try:
            pd.to_numeric(column)
            quantitative_columns.append(column_name)
            df[column_name].fillna(0, inplace=True)
        except ValueError:
            qualitative_columns.append(column_name)
            df[column_name].replace({pd.NA: 'NA'}, inplace=True)

#Conteggio dei NaN per ogni attributo
nan_counts = df.isna().sum()/len(df)

#%%

# =============================================================================
# Correzione del problema Garage e Basement;
# =============================================================================

Garage_index = nan_counts[nan_counts == nan_counts['GarageFinish']].index
replace_nan_values(df, Garage_index)
replace_nan_values(test_df, Garage_index)


Bsmt_index = nan_counts[nan_counts == nan_counts['BsmtQual']].index
replace_nan_values(df, Bsmt_index)
replace_nan_values(test_df, Bsmt_index)


Bsmt_index = nan_counts[nan_counts == nan_counts['BsmtFinType2']].index
replace_nan_values(df, Bsmt_index)
replace_nan_values(test_df, Bsmt_index)
# df[Bsmt_index] = df[Bsmt_index].fillna(0)
# test_df[Bsmt_index] = test_df[Bsmt_index].fillna(0)

# =============================================================================
#  Sostituzione Di Electrical con SBrkr
# =============================================================================
 
df['Electrical'] = df['Electrical'].fillna('Sbrkr')
test_df['Electrical'] = test_df['Electrical'].fillna('Sbrkr')


# =============================================================================
# Correzione di MasVnr Type ed Area Assumo che non ve ne sia nessuno nei nan
# =============================================================================

df['MasVnrType'] = df['MasVnrType'].fillna('none')
test_df['MasVnrType'] = test_df['MasVnrType'].fillna('none')

df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(0)


# =============================================================================
# Correzione del NaN nel Fireplace Quando in realta e' mancante
# =============================================================================

df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna('NA')

# =============================================================================
# Eliminazione MOMENTANEA DELLACOLONNA FRONTE STRADALE
# =============================================================================

del df['LotFrontage']
del test_df['LotFrontage']


# =============================================================================
# Eliminazione della Colonne con tanti NaN in percentuale
# =============================================================================

del df['Alley']
del test_df['Alley']
del df['PoolQC']
del test_df['PoolQC']
del df['Fence']
del test_df['Fence']
del df['MiscFeature']
del test_df['MiscFeature']


nan_counts = df.isna().sum()/len(df)
#%%
# =============================================================================
# Eliminazione della Colonne perche tutti uguali
# =============================================================================

del df['Street']
del test_df['Street']
del df['Utilities']
del test_df['Utilities']
del df['MiscVal']
del test_df['MiscVal']


#%%

# =============================================================================
# Eliminazione ad occhio
# =============================================================================


for i in df.index:
    if df.loc[i, 'Exterior1st'] != df.loc[i, 'Exterior2nd']:
        df.loc[i, 'Exterior1st'] = df.loc[i, 'Exterior1st'] + df.loc[i, 'Exterior2nd']

for i in test_df.index:
    if test_df.loc[i, 'Exterior1st'] != test_df.loc[i, 'Exterior2nd']:
        test_df.loc[i, 'Exterior1st'] = test_df.loc[i, 'Exterior1st'] + test_df.loc[i, 'Exterior2nd']
        
del df['Exterior2nd']
del test_df['Exterior2nd'] #Merged sopra

#merge dei Brk Cmn

del df['Heating']
del test_df['Heating'] 

del df['TotalBsmtSF']
del test_df['TotalBsmtSF'] # ci sono tutti gli addendi

#%%
#SICURAMENTE INUTILI


del df['GarageCond']
del test_df['GarageCond'] #GarageQual e uguale (df['GarageCond'] == df['GarageQual']).sum() Magari fare una media, aggiungi quello mancante negli encoding numerici
del df['RoofMatl']
del test_df['RoofMatl'] # tutto dello stesso valore vedi entropia e distribuzione
del df['Condition2']
del test_df['Condition2'] # tutto dello stesso valore vedi entropia e distribuzione
del df['Electrical']
del test_df['Electrical']
#%%
del df['PavedDrive']
del test_df['PavedDrive'] #poca varianza
del df['CentralAir']
del test_df['CentralAir']
del df['LandSlope']
del test_df['LandSlope']

#%%

# Mappatura per le colonne Qualitative
mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
columns_to_replace = [
    'PoolQC', 'GarageQual', 'GarageCond', 'FireplaceQu', 
    'KitchenQual', 'HeatingQC', 'BsmtCond', 'BsmtQual', 
    'ExterCond', 'ExterQual'
]
df = replace_qualitative_values(df, columns_to_replace, mapping)
test_df = replace_qualitative_values(test_df, columns_to_replace, mapping)

# Mappatura per le colonne BsmtFinType1 e BsmtFinType2
mapping = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
columns_to_replace = ['BsmtFinType1', 'BsmtFinType2']
df = replace_qualitative_values(df, columns_to_replace, mapping)
test_df = replace_qualitative_values(test_df, columns_to_replace, mapping)

# Mappatura per la colonna BsmtExposure
mapping = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
columns_to_replace = ['BsmtExposure']
df = replace_qualitative_values(df, columns_to_replace, mapping)
test_df = replace_qualitative_values(test_df, columns_to_replace, mapping)

# Mappatura per la colonna GarageFinish
mapping = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
columns_to_replace = ['GarageFinish']
df = replace_qualitative_values(df, columns_to_replace, mapping)
test_df = replace_qualitative_values(test_df, columns_to_replace, mapping)

# Mappatura per la colonna Functional
mapping = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}
columns_to_replace = ['Functional']
df = replace_qualitative_values(df, columns_to_replace, mapping)
test_df = replace_qualitative_values(test_df, columns_to_replace, mapping)

#%%

# =============================================================================
# Estrazione della lista di feature categoriche e quantitative
# =============================================================================

quantitative_cols = []
qualitative_cols = []

for column_name in df.columns:
    column = df[column_name]
    try:
        pd.to_numeric(column)
        quantitative_cols.append(column_name)

    except ValueError:
        qualitative_cols.append(column_name)


quantitative_cols = quantitative_cols[2:-1] #Tolgo la colonna relativa alla tipologia di casa, Unnamed e la Label

qualitative_cols.append('MSSubClass')

quantitative_df = df[quantitative_cols] #creazione dataset feature quantitative



#%% Plotting Categoriche e Calcoli entropie 
#Le features da eliminare secondo questa analisi sono state gia eliminate 
#nelle sezioni di codice precedente, qui si riporta  per completezza il codice
#dei generazione degli istogrammi e di calcolo dell'entropia
# =============================================================================
# Plot delle categoriche
# =============================================================================

for i in qualitative_cols:
    contingency_table = pd.crosstab(df[i], df['Label'])
    contingency_table.plot(kind='bar', stacked=True)
    plt.show()
    


# =============================================================================
# Calcolo delle entropie
# =============================================================================
from scipy.stats import entropy

entropies = {}
for col in df[qualitative_cols].columns:
    values_counts = df[col].value_counts(normalize=True)  # Calcola le frequenze relative
    probabilities = values_counts.values
    entropies[col] = entropy(probabilities, base=2)  # Calcolo dell'entropia

print("Entropia di ciascuna colonna:")
for col, ent in entropies.items():
    print(f"{col}: {ent}")

#%%
# =============================================================================
# TEST CHI2 su feature categoriche
# =============================================================================


qualitative_cols = df.select_dtypes(include=['object']).columns


chi2_corr_mat = pd.DataFrame(np.nan, index=qualitative_cols, columns=qualitative_cols)

# Calcola i valori p del test chi-quadro per ogni coppia di variabili categoriche
for k in qualitative_cols:
    for j in qualitative_cols:
        if k != j:
            res = pd.crosstab(df[k], df[j])
            stat, p, dof, expected = chi2_contingency(res)
            chi2_corr_mat.loc[k, j] = p

# Converte i valori p in numerici
chi2_corr_mat = chi2_corr_mat.astype(float)

# =============================================================================
# PLOT TEST CHI2
# =============================================================================

fig, ax = plt.subplots(figsize=(15, 15))
cax = ax.matshow(chi2_corr_mat, cmap='coolwarm', vmin=0, vmax=1)
plt.xticks(range(len(chi2_corr_mat.columns)), chi2_corr_mat.columns, rotation=45, ha='left')
plt.yticks(range(len(chi2_corr_mat.index)), chi2_corr_mat.index)
plt.colorbar(cax)
plt.show()

#%% Creazione di dataset X,y e X_test,y_test
# =============================================================================
# Creazione di dataset X,y e X_test,y_test
# =============================================================================

X = df.drop(columns=['Unnamed: 0', df.columns[-1]], inplace=False)
y = df[df.columns[-1]]

X_test = test_df.drop(columns=['Unnamed: 0', df.columns[-1]], inplace=False)
y_test = test_df[test_df.columns[-1]]

#%% Standard SCALING PER DBSCAN

# =============================================================================
#                               Standard SCALING PER DBSCAN & LDA
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X[quantitative_cols] = scaler.fit_transform(X[quantitative_cols])
X_test[quantitative_cols] = scaler.transform(X_test[quantitative_cols])




#%% DBSCAN & LDA 3D INTRACLASSE

# =============================================================================
#                               DBSCAN & LDA 3D INTRACLASSE
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from plotly.offline import plot



# Eseguiamo LDA per ottenere le componenti principali
lda = LDA(n_components=3)
X_lda = lda.fit_transform(X[quantitative_cols], y)

df_lda_and_class = np.hstack([X_lda, y.to_frame()])

fig = go.Figure()

#Primo plot di tutto il dataset

fig.add_trace(go.Scatter3d(
    x=X_lda[:, 0],
    y=X_lda[:, 1],
    z=X_lda[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=y,  
        colorscale='Viridis',  
        opacity=0.8
    ),
    name='Points'
))

# Setup del plot
fig.update_layout(
    scene=dict(
        xaxis_title='LD1',
        yaxis_title='LD2',
        zaxis_title='LD3'
    ),
    title='LDA in 3D'
)


plot(fig) 

#Analisi intraclasse e singoli plots

# Utilizziamo DBSCAN per trovare gli outliers ciclicamente su ogni classe
eps = [0.8, 0.8, 0.8, 0.8]
min_samples = [5, 5, 6, 6]

dbscan = DBSCAN(eps=eps, min_samples=min_samples)

rows_to_keep = np.ones(len(df_lda_and_class), dtype=bool)

for i in range(4):
    
    dbscan = DBSCAN(eps=eps[i], min_samples=min_samples[i])
    
    ith_clas_df = df_lda_and_class[df_lda_and_class[:, -1] == i]

    labels = dbscan.fit_predict(ith_clas_df)

    
    outlier_mask = labels == -1
    inlier_mask = labels != -1

    
    fig = go.Figure()

    # Inliers
    fig.add_trace(go.Scatter3d(
        x=ith_clas_df[inlier_mask, 0],
        y=ith_clas_df[inlier_mask, 1],
        z=ith_clas_df[inlier_mask, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=ith_clas_df[inlier_mask, 3],  
            colorscale='Viridis',  
            opacity=0.8
        ),
        name='Inliers' + str(sum(inlier_mask)) + ' C:' + str(y[i])
    ))
    
    # Aggiungiamo gli outlier
    fig.add_trace(go.Scatter3d(
        x=ith_clas_df[outlier_mask, 0],
        y=ith_clas_df[outlier_mask, 1],
        z=ith_clas_df[outlier_mask, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='red',           
            opacity=0.8
        ),
        name='Outliers' + str(sum(outlier_mask))
    ))

    
    fig.update_layout(
        scene=dict(
            xaxis_title='LD1',
            yaxis_title='LD2',
            zaxis_title='LD3'
        ),
        title='LDA in 3D with Outliers Highlighted'
    )
    
    
    #plot(fig) Decommentare per avere il plot sul browser
    fig.write_html("C:/Users/Loren/Documents/Polito/Business Intelligence/classe"+str(y[i])+".html")

    # Filtriamo gli outliers dal dataframe principale
    class_indices = np.where(df_lda_and_class[:, -1] == i)[0]
    rows_to_keep[class_indices[outlier_mask]] = False
#%%
# DataFrame principale senza gli outliers
X = X[rows_to_keep]
y = y[rows_to_keep]

print(f"Shape of the filtered dataframe: {X.shape}")
#%% LDA PER FEATURES
# =============================================================================
#                        LDA PER FEATURE SELECTION
# =============================================================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


lda = LinearDiscriminantAnalysis(n_components=3)
X_lda = lda.fit_transform(X[quantitative_cols], y)

lda_coefficients = lda.coef_

#%%
# Dizionario per memorizzare le migliori feature per ogni classe
best_features_per_class = {}

x = 20
for class_label in np.unique(y):
    # Trova le migliori x feature per la classe corrente
    class_coeffs = lda_coefficients[class_label]
    top_features_idx = np.argsort(np.abs(class_coeffs))[::-1][:x]  
    top_features = X[quantitative_cols].columns[top_features_idx]  
    best_features_per_class[class_label] = top_features.tolist()


for class_label, top_features in best_features_per_class.items():
    print(f"Classe {class_label}: {top_features}")
    
quantitative_finali = []
for class_label, top_features in best_features_per_class.items():
    quantitative_finali.extend(top_features)

quantitative_finali = list(set(quantitative_finali))



#%% ENCODING delle feature categoriche non ordinabili
# =============================================================================
# ENCODING delle feature categoriche non ordinabili
# =============================================================================

#categorical_cols = [col for col in X.columns if col not in quantitative_cols]


to_dummy = pd.concat([X, X_test], axis = 0)

df_categorical = pd.get_dummies(to_dummy[qualitative_cols])
to_dummy = pd.concat([to_dummy[quantitative_cols], df_categorical], axis=1)

X = to_dummy.iloc[:X.shape[0]]
X_test = to_dummy.iloc[X.shape[0]:]

qualitative_encoded_cols = df_categorical.columns



#%% N- FEATURE SELECTION - FEATURES PERMUTATION
# =============================================================================
#                             FEATURES PERMUTATION
# =============================================================================
from sklearn.model_selection import GridSearchCV 
  
X_train, X_tune, y_train, y_tune = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)

# defining parameter range 
param_grid = {'bootstrap': [True],
 'max_depth': [ 10, 20, 30],
 'max_features': ['log2', 'sqrt'],
 'n_estimators': [ 300, 400],
 'criterion': ['entropy', 'gini'],
 'random_state': [42]}
  
grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 4) 
  
# fitting the model for grid search 
grid.fit(X_train[qualitative_encoded_cols], y_train) 

model = grid.best_estimator_
y_pred = model.predict(X_tune[qualitative_encoded_cols])
#accuracy = accuracy_score(y_pred, y_tune)
#%
start_time = time.time()
result = permutation_importance(
    model, X_tune[qualitative_encoded_cols], y_tune, n_repeats=600, random_state=37, n_jobs=-1, 
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


#%%

forest_importances = pd.Series(result.importances_mean, index=X_train[qualitative_encoded_cols].columns)
forest_importances = forest_importances.sort_values()

#%% Soglia con cui si selezionano le migliori features categoriche
categoriche_finali = forest_importances[forest_importances > 0.0003].index
#forest_importances.set_index('Unnamed: 0', inplace=True)
#%% CREAZIONE DEL DATASET CON COLONNE DI LDA E FEATURES SELEZIONATE PRECEDENTEMENTE
X_lda = pd.DataFrame(X_lda, columns=['LDA_1', 'LDA_2', 'LDA_3'])
X = X.reset_index(drop = True)
X = pd.concat([X, X_lda], axis=1)

X_test_lda = lda.transform(X_test[quantitative_cols])
X_test_lda = pd.DataFrame(X_test_lda, columns=['LDA_1', 'LDA_2', 'LDA_3'])
X_test = pd.concat([X_test, X_test_lda], axis=1)

#%%
colonne_finali = quantitative_finali + categoriche_finali.tolist() + ['LDA_1', 'LDA_2', 'LDA_3']

#%% CLASSIFICAZIONE
# =============================================================================
#                               CLASSIFICAZIONE
# =============================================================================


#%% RandomForest
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV 
  
param_dist = {
    'n_estimators': randint(10, 1000), 
    'criterion': ['gini', 'entropy'],  
    'max_depth': [None, 10, 20, 25, 30, 50],  
    'min_samples_split': randint(2, 50), 
    'min_samples_leaf': randint(1, 15), 
    'max_features': ['auto', 'sqrt', 'log2'],  
    'bootstrap': [True, False]  
}


random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=1000,  
    scoring='accuracy',  
    cv=5,  
    verbose=1,
    random_state=42,
    n_jobs=-1 
)


random_search.fit(X[colonne_finali], y) 

model_RF = random_search.best_estimator_

y_pred = model_RF.predict(X_test[colonne_finali])
accuracy_RF = accuracy_score(y_test, y_pred)



#%% SVC
from sklearn.model_selection import GridSearchCV 
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV

param_dist = {
    'C': uniform(0.1, 100),  
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'degree': randint(2, 10),  
    'gamma': ['scale', 'auto'] + list(uniform(0.001, 10).rvs(10)),  
    'coef0': uniform(0, 10), 
    'class_weight': [None, 'balanced']
}



random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_dist,
    n_iter=1000,  
    scoring='accuracy',  
    cv=5,  
    refit = True,
    verbose=4,
    random_state=42,
    n_jobs=-1  
)
  

random_search.fit(X[colonne_finali], y) 

model_SVC = random_search.best_estimator_

y_pred = model_SVC.predict(X_test[colonne_finali])
accuracy_SVC = accuracy_score(y_test, y_pred)


#%% KNN

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint


knn = KNeighborsClassifier()


param_dist = {
    'n_neighbors': randint(1, 100),  
    'weights': ['uniform', 'distance'], 
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  
    'leaf_size': randint(10, 50),  
    'metric': ['euclidean', 'manhattan', 'chebyshev'],
    'p': [1, 2, 3] 
}


random_search = RandomizedSearchCV(
    estimator=knn,
    param_distributions=param_dist,
    n_iter=10000,  # Numero di iterazioni
    scoring='accuracy',  
    cv=5,  
    verbose=1,
    random_state=42,
    n_jobs=-1 
)

random_search.fit(X[colonne_finali], y) 

model_KNN = random_search.best_estimator_

y_pred = model_KNN.predict(X_test[colonne_finali])
accuracy_KNN = accuracy_score(y_test, y_pred)


#%% Confusion Matrix

#Script usato per calcolare la confusion matrix dei modelli quando necessario
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


plt.show()

