from Model.PreprocesadoTexto import _normalizarTexto
from sklearn.feature_extraction.text import TfidfVectorizer

texto = 'Esto es un tweet dë ejemplô@elvergalarga còn cárâr.#como dije.(te crees)'
print('Sin normalizar ', texto)
print('Normalizado ',_normalizarTexto(texto))

bv = TfidfVectorizer(ngram_range=(1,2), min_df=2, norm=None)

