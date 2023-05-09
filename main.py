from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from Model.CargarDatos import cargarDatos
from Model.PreprocesadoTexto import normalizarDoc
from sklearn.model_selection import train_test_split
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
    
    tfidfVectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,2), min_df=25)
    tfidf_train = tfidfVectorizer.fit_transform(train_corpus)
    tfidf_test = tfidfVectorizer.transform(test_corpus)
    y_train_sparse = sparse.csr_matrix(y_train)
    y_test_sparse = sparse.csr_matrix(y_test)
    print('X_train dimension= ', tfidf_train.shape)
    print('X_test dimension= ', tfidf_test.shape)
    print('y_train dimension= ', y_train_sparse.shape)
    print('y_train dimension= ', y_test_sparse.shape)
    
    modelLR = LogisticRegression(multi_class='ovr',solver='liblinear')
    clf = MultiOutputClassifier(estimator=modelLR)
    

    clf.fit(tfidf_train, y_train_sparse.toarray())
    y_pred = clf.predict(tfidf_test)
    metrica = get_metrics(true_labels=y_test, predicted_labels=y_pred)
    data = pd.DataFrame([('xD', metrica['Accuracy'], metrica['F1 Score'], metrica['Precision'],metrica['Recall'])], columns=['Modelo', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
    print(data)
    '''modelLR = LogisticRegression(multi_class='ovr',solver='liblinear')
    prediccion, metrica = train_predict_evaluate_model(classifier=modelLR,
                                                       train_features=tfidf_train.toarray(),
                                                       train_labels=y_train_bin,
                                                       test_features=tfidf_test.toarray(),
                                                       test_labels=y_test_bin)
    data = pd.DataFrame([('xD', metrica['Accuracy'], metrica['F1 Score'], metrica['Precision'],metrica['Recall'])], columns=['Modelo', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
    print(data)''' 

if __name__ == "__main__":
    main()
