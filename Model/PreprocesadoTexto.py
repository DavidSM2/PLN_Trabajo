import re
import spacy
from textacy import preprocessing


nlp = spacy.load("es_core_news_lg")

acentos = [
    ('á', 'a'), ('à', 'a'), ('ä', 'a'), ('â', 'a'),
    ('é', 'e'), ('è', 'e'), ('ë', 'e'), ('ê', 'e'),
    ('í', 'i'), ('ì', 'i'), ('ï', 'i'), ('î', 'i'),
    ('ó', 'o'), ('ò', 'o'), ('ö', 'o'), ('ô', 'o'),
    ('ú', 'u'), ('ù', 'u'), ('ü', 'u'), ('û', 'u')
]
   


def normalizarDoc(doc):
    doc_norm = []
    for tweet in doc:
        tweet_norm = _normalizarTexto(tweet)
        doc_norm.append(tweet_norm)
    return doc_norm


def _normalizarTexto(texto):
    # separamos después de ciertos signos de puntuación
    texto = _quitarSignos(texto)
    texto = _quitarNumeros(texto)
    #texto = _normalizarUnicodeWhitespace(texto)
    
    #texto = _normalizarCaracteristicasTwets(texto) #Esto no aporta mejoras
    doc = nlp(texto)
    tokens = [t.lemma_.lower()
              for t in doc if not t.is_punct and not t.is_space and len(t.text) > 2]
    salida = ' '.join(tokens)
    #print(salida)
    return salida

def _normalizarUnicodeWhitespace(texto):
    texto = preprocessing.normalize.whitespace(texto)
    texto = preprocessing.normalize.unicode(texto)
    return texto
def _quitarSignos(texto):
    texto = re.sub(r"([\.\?])", r"\1 ", texto)
    return texto

def _quitarNumeros(texto):
    texto = re.sub(r'(\d+|\n)','',texto)
    return texto

def _quitarAcentos(texto):
    for letraAccent, letraNorm in acentos:
        texto = re.sub(r'{0}'.format(letraAccent), letraNorm, texto)
    return texto
def _normalizarPuncts(texto):
    texto = re.sub(r'(?P<punct>[\.\,])(?P<simbol>[\@\#\¿\?\¡\!\(\)])', r'\g<punct> \g<simbol>', texto)
    return texto
def _normalizarCaracteristicasTwets(texto):
    texto = preprocessing.replace.hashtags(texto, repl='HASHTAGS')
    '''texto = preprocessing.replace.emails(texto, repl='EMAIL')
    texto = preprocessing.replace.emojis(texto, repl='EMOJI')
    texto = preprocessing.replace.urls(texto, repl='URL')'''
    texto = preprocessing.replace.user_handles(texto, repl='USUARIO')
    return texto
