from Model.PreprocesadoTexto import _normalizarTexto

texto = 'Esto es un tweet dë ejemplô@elvergalarga còn cárâr.#como dije.(te crees)'
print('Sin normalizar ', texto)
print('Normalizado ',_normalizarTexto(texto))