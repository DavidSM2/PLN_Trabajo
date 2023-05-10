from scipy import sparse
from sklearn.calibration import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from Model.CargarDatos import cargarDatos
from Model.PreprocesadoTexto import normalizarDoc
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
import pandas as pd

def get_metrics(true_labels, predicted_labels):
    """Calculamos distintas métricas sobre el
    rendimiento del modelo. Devuelve un diccionario
    con los parámetros medidos"""

    return {
        'Accuracy': np.round(
            metrics.accuracy_score(true_labels,
                                   predicted_labels),
            3),
        'Precision': np.round(
            metrics.precision_score(true_labels,
                                    predicted_labels,
                                    average='weighted',
                                    zero_division=0),
            3),
        'Recall': np.round(
            metrics.recall_score(true_labels,
                                 predicted_labels,
                                 average='weighted',
                                 zero_division=0),
            3),
        'F1 Score': np.round(
            metrics.f1_score(true_labels,
                             predicted_labels,
                             average='weighted',
                             zero_division=0),
            3)}


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    """Función que entrena un modelo de clasificación sobre
    un conjunto de entrenamiento, lo aplica sobre un conjunto
    de test y devuelve la predicción sobre el conjunto de test
    y las métricas de rendimiento"""
    # genera modelo
    classifier.fit(train_features, train_labels)
    # predice usando el modelo sobre test
    predictions = classifier.predict(test_features)
    # evalúa rendimiento de la predicción
    metricas = get_metrics(true_labels=test_labels,
                           predicted_labels=predictions)
    return predictions, metricas

def ajusteHiperparametros(param_grid, model, x_train, y_train, scoring):
    # Búsqueda por validación cruzada
    # ==============================================================================

    grid = GridSearchCV(
            estimator  = model,
            param_grid = param_grid,
            scoring    = 'accuracy',
            n_jobs     = -1,
            cv         = 5, 
            verbose    = 0,
            return_train_score = True
        )

    # Se asigna el resultado a _ para que no se imprima por pantalla
    _ = grid.fit(X = x_train, y = y_train)

    # Resultados del grid
    # ==============================================================================
    resultados = pd.DataFrame(grid.cv_results_)
    filter = resultados.filter(regex = '(param.*|mean_t|std_t)')\
        .drop(columns = 'params')\
        .sort_values('mean_test_score', ascending = False) \
        .head(5)
    print(filter)
    
def main():
    dfCsv = cargarDatos('./Datos_test/sem_eval_train_es.csv')
    corpus = list(dfCsv['Tweet'])
    corpus = normalizarDoc(corpus)
    # print(arrayTweets)
    dfTags = dfCsv.drop('ID', axis=1).drop('Tweet', axis=1)
    labels = dfTags.to_numpy()
    # print(arrayTags)
    train_corpus, test_corpus, y_train, y_test = train_test_split(corpus,
                                                                  labels,
                                                                  test_size=0.3,
                                                                  random_state=0
                                                                  )
    
    tfidfVectorizer = TfidfVectorizer()
    tfidf_train = tfidfVectorizer.fit_transform(train_corpus)
    tfidf_test = tfidfVectorizer.transform(test_corpus)
    y_train_sparse = sparse.csr_matrix(y_train)
    
    modelSvc = LinearSVC(C=0.7,  
                         penalty='l1', 
                         dual=False)
    
    clf = MultiOutputClassifier(estimator=modelSvc)
    

    clf.fit(tfidf_train, y_train_sparse.toarray())
    y_pred = clf.predict(tfidf_test)
    metrica = get_metrics(true_labels=y_test, predicted_labels=y_pred)
    data = pd.DataFrame([("multi_class='ovr'", metrica['Accuracy'], metrica['F1 Score'], metrica['Precision'],metrica['Recall'])], columns=['Modelo', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
    print(data)
# ==============================================================================
    # Grid de hiperparámetros
    #Con este metodo ajustamos los hiperparametros
    '''param_grid = {
        'C': [0.1, 0.5, 1, 5],
        'multi_class': ['ovr', 'crammer_singer'],
        'max_iter': [500, 1000, 2000]
    }
    ajusteHiperparametros(param_grid=param_grid,
                            model=modelSvc,
                            x_train=tfidf_train,
                            y_train=y_train_sparse.toarray(),
                            scoring='accuracy')'''
# ==============================================================================  

if __name__ == "__main__":
    main()
