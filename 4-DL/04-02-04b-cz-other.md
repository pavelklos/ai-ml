# Word Embeddings a Jazykové Modely

## Obsah
1. Úvod do Word Embeddings
2. Typy Word Embeddings
3. Word2Vec
4. GloVe
5. FastText
6. Kontextové embeddingy (ELMO, BERT)
7. Jazykové modely
8. n-gram jazykové modely
9. Neuronové jazykové modely
10. Transformery a pokročilé jazykové modely
11. Vyhodnocování jazykových modelů
12. Praktické aplikace
13. Shrnutí

## Úvod do Word Embeddings

Word embeddings (česky vektorové reprezentace slov) jsou způsoby reprezentace slov jako vektorů v mnohorozměrném prostoru. Na rozdíl od tradičních metod, jako je one-hot encoding, zachycují word embeddings sémantické a syntaktické vztahy mezi slovy, kde podobná slova mají podobné vektory.

**Klíčové vlastnosti word embeddings:**
- Reprezentují slova jako husté vektory reálných čísel
- Zachycují sémantické vztahy mezi slovy
- Umožňují matematické operace se slovy (např. "král" - "muž" + "žena" ≈ "královna")
- Výrazně snižují dimenzionalitu oproti one-hot encodingu
- Jsou základním stavebním prvkem pro NLP úlohy

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ukázka vektorové reprezentace slov (zjednodušená)
word_vectors = {
    "pes": np.array([0.2, 0.4, 0.7, 0.1]),
    "kočka": np.array([0.3, 0.5, 0.6, 0.2]),
    "auto": np.array([0.9, 0.1, 0.2, 0.4]),
    "kolo": np.array([0.8, 0.2, 0.1, 0.5])
}

# Vizualizace vektorů pomocí PCA pro redukci dimenze
vectors = np.array(list(word_vectors.values()))
words = list(word_vectors.keys())

# Redukce dimenze na 2D pro vizualizaci
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Vykreslení vektorů
plt.figure(figsize=(8, 6))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], marker='o')

# Přidání popisků slov
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=12)

plt.title("Vizualizace word embeddings pomocí PCA")
plt.xlabel("Komponenta 1")
plt.ylabel("Komponenta 2")
plt.grid(True)
plt.show()
```

## Typy Word Embeddings

Existuje několik způsobů vytváření vektorových reprezentací slov:

1. **Statické embeddingy** - jeden vektor pro každé slovo bez ohledu na kontext
   - Word2Vec, GloVe, FastText

2. **Kontextové embeddingy** - vektory slov závisí na kontextu v rámci věty
   - ELMo, BERT

3. **Metody založené na četnosti** - LSA, LDA

4. **Pretrénované embeddingy** - vektory natrénované na velkých korpusech a připravené k použití

## Word2Vec

Word2Vec je jedna z nejpopulárnějších metod pro tvorbu word embeddings, vyvinutá Tomášem Mikolovem a týmem Google v roce 2013. Existují dvě hlavní architektury:

1. **Continuous Bag of Words (CBOW)** - předpovídá cílové slovo z okolního kontextu
2. **Skip-gram** - předpovídá okolní kontext ze slova

```python
import gensim.downloader as api
from gensim.models import Word2Vec
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Stažení vzorových textových dat
nltk.download('brown')
from nltk.corpus import brown

# Příprava dat
sentences = brown.sents()
print(f"Počet vět v datasetu: {len(sentences)}")
print(f"Ukázka věty: {sentences[0]}")

# Trénování modelu Word2Vec
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4)

# Uložení a načtení modelu
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

# Získání vektoru pro konkrétní slovo
vector = model.wv['man']
print(f"Vektor pro slovo 'man' (prvních 5 hodnot): {vector[:5]}")

# Nalezení podobných slov
similar_words = model.wv.most_similar('man', topn=5)
print("Slova podobná slovu 'man':")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# Sémantické operace
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"king - man + woman = {result[0][0]} (skóre: {result[0][1]:.4f})")

# Vizualizace vybraných slov pomocí PCA
def plot_words(model, words):
    # Získání vektorů pro vybraná slova
    word_vectors = [model.wv[word] for word in words]
    
    # Redukce dimenze pomocí PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(word_vectors)
    
    # Vykreslení
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c='steelblue', edgecolors='k')
    
    # Přidání popisků slov
    for i, word in enumerate(words):
        plt.annotate(word, (pca_result[i, 0], pca_result[i, 1]), fontsize=12)
    
    plt.title("Vizualizace word embeddings")
    plt.grid(True)
    plt.show()

# Vizualizace vybraných slov
words_to_plot = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 
                'father', 'mother', 'doctor', 'nurse', 'programmer', 'teacher']
plot_words(model, words_to_plot)
```

### Použití předtrénovaných Word2Vec modelů

```python
import gensim.downloader as api

# Dostupné předtrénované modely
print(api.info())

# Načtení předtrénovaného modelu
word2vec_model = api.load('word2vec-google-news-300')

# Získání podobných slov
similar_words = word2vec_model.most_similar('python', topn=10)
print("Slova podobná slovu 'python':")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")
```

## GloVe

GloVe (Global Vectors for Word Representation) je další populární metoda pro tvorbu word embeddings, vyvinutá na Stanfordské univerzitě. Na rozdíl od Word2Vec, který je prediktivním modelem, je GloVe založen na počítání společných výskytů slov v korpusu.

```python
import numpy as np
from scipy import spatial
import requests
import zipfile
import io
import os

# Stažení předtrénovaných GloVe vektorů
def download_glove_vectors(url='https://nlp.stanford.edu/data/glove.6B.zip', 
                           dest_folder='data'):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Kontrola, zda už soubory existují
    if os.path.exists(os.path.join(dest_folder, 'glove.6B.100d.txt')):
        print("GloVe vektory již existují.")
        return
    
    print(f"Stahuji GloVe vektory z {url}...")
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(dest_folder)
    print("Stažení dokončeno!")

# Načtení GloVe vektorů do slovníku
def load_glove_vectors(filename):
    embeddings_dict = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    
    return embeddings_dict

# Stažení a načtení vektorů
download_glove_vectors()
embeddings_dict = load_glove_vectors('data/glove.6B.100d.txt')

# Nalezení nejpodobnějších slov
def find_closest_words(embedding_dict, word, n=5):
    word_vector = embedding_dict.get(word)
    
    if word_vector is None:
        return []
    
    # Výpočet podobnosti mezi vektorem slova a všemi ostatními vektory
    similarities = {}
    for key, vector in embedding_dict.items():
        if key == word:
            continue
        similarities[key] = 1 - spatial.distance.cosine(word_vector, vector)
    
    # Seřazení podle podobnosti
    closest_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
    return closest_words

# Příklady podobných slov
word = 'computer'
similar_words = find_closest_words(embeddings_dict, word)

print(f"Slova podobná slovu '{word}':")
for similar_word, similarity in similar_words:
    print(f"  {similar_word}: {similarity:.4f}")

# Sémantické operace s vektory
def vector_operation(positive_words, negative_words, embedding_dict, n=1):
    # Výpočet výsledného vektoru
    result_vector = np.zeros(next(iter(embedding_dict.values())).shape)
    
    for word in positive_words:
        if word in embedding_dict:
            result_vector += embedding_dict[word]
    
    for word in negative_words:
        if word in embedding_dict:
            result_vector -= embedding_dict[word]
    
    # Nalezení nejpodobnějších slov k výslednému vektoru
    similarities = {}
    for key, vector in embedding_dict.items():
        if key in positive_words or key in negative_words:
            continue
        similarities[key] = 1 - spatial.distance.cosine(result_vector, vector)
    
    closest_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
    return closest_words

# Příklad: king - man + woman = ?
result = vector_operation(['king', 'woman'], ['man'], embeddings_dict)
print(f"king - man + woman = {result[0][0]} (skóre: {result[0][1]:.4f})")
```

## FastText

FastText, vyvinutý týmem Facebook AI Research, je rozšíření modelu Word2Vec, které pracuje na úrovni subword jednotek (n-gramů znaků). To umožňuje modelu generovat embeddingy i pro slova, která nebyla v trénovacím datasetu.

```python
import fasttext
import fasttext.util
import tempfile
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Stažení předtrénovaného modelu FastText (zmenšená verze)
fasttext.util.download_model('cs', if_exists='ignore')

# Načtení modelu
ft_model = fasttext.load_model('cc.cs.300.bin')

# Získání vektoru pro slovo
word = "počítač"
vector = ft_model.get_word_vector(word)
print(f"Vektor pro slovo '{word}' (prvních 5 hodnot): {vector[:5]}")

# Ukázka práce s OOV (out-of-vocabulary) slovy
made_up_word = "superpočítačový"  # FastText dokáže generovat vektory i pro neviděná slova
vector = ft_model.get_word_vector(made_up_word)
print(f"Vektor pro neviděné slovo '{made_up_word}' (prvních 5 hodnot): {vector[:5]}")

# Vizualizace sémantických vztahů pomocí PCA
def visualize_word_embeddings(model, words):
    # Získání vektorů
    vectors = np.array([model.get_word_vector(word) for word in words])
    
    # Redukce dimenze pomocí PCA
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)
    
    # Vizualizace
    plt.figure(figsize=(12, 8))
    plt.scatter(result[:, 0], result[:, 1], c='steelblue')
    
    # Přidání popisků slov
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=12)
    
    plt.title("2D projekce word embeddings pomocí PCA")
    plt.grid(True)
    plt.show()

# Příklad vizualizace
czech_words = ["pes", "kočka", "kůň", "slon", "jablko", "banán", "pomeranč", 
              "hruška", "auto", "letadlo", "vlak", "loď"]
visualize_word_embeddings(ft_model, czech_words)

# Trénování vlastního FastText modelu na vlastních datech
# Příprava cvičných dat
sample_text = [
    "python je vysokoúrovňový programovací jazyk",
    "tensorflow je knihovna pro strojové učení",
    "pytorch je populární framework pro deep learning",
    "programování je důležité pro vývoj softwaru",
    "strojové učení je součást umělé inteligence"
]

# Uložení dat do dočasného souboru
temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
for line in sample_text:
    temp_file.write(line + "\n")
temp_file.close()

# Trénování modelu
model = fasttext.train_unsupervised(temp_file.name, model='skipgram')

# Uložení a načtení modelu
model.save_model("model_czech.bin")
model = fasttext.load_model("model_czech.bin")

# Odstranění dočasného souboru
os.unlink(temp_file.name)

# Test natrénovaného modelu
words = ["python", "programování", "učení"]
for word in words:
    print(f"\nSlova podobná slovu '{word}':")
    similar_words = model.get_nearest_neighbors(word, k=3)
    for score, similar_word in similar_words:
        print(f"  {similar_word}: {score:.4f}")
```

## Kontextové embeddingy

Na rozdíl od statických embeddings, jako jsou Word2Vec nebo GloVe, kontextové embeddings generují různé vektorové reprezentace stejného slova v závislosti na jeho kontextu v rámci věty.

### ELMo (Embeddings from Language Models)

ELMo reprezentace jsou založeny na biLSTM jazykovém modelu, kde embedding slova je funkcí celé věty, ve které se slovo nachází.

```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Načtení ELMo modelu z TensorFlow Hub
elmo = hub.load("https://tfhub.dev/google/elmo/3")

# Funkce pro získání ELMo embeddings
def get_elmo_embeddings(texts):
    embeddings = elmo.signatures["default"](
        tf.constant(texts))["elmo"]
    return embeddings.numpy()

# Příklad vět pro ukázku kontextových embeddings
sentences = [
    "Jablko spadlo ze stromu.",
    "Apple představil nový iPhone.",
    "Steve Jobs založil Apple.",
    "Mám rád jablkový koláč."
]

# Získání embeddings pro všechny věty
embeddings = get_elmo_embeddings(sentences)

# Výpis tvaru embeddings
print(f"Tvar ELMo embeddings: {embeddings.shape}")
# Výstup bude (počet_vět, max_délka_věty, dimenze_embeddings)

# Získání embeddings pro konkrétní slovo v různých kontextech
# Identifikujme pozice slov "Apple" a "jablko" v každé větě
word_indices = {
    0: 0,  # "Jablko" v první větě je na pozici 0
    1: 0,  # "Apple" v druhé větě je na pozici 0
    2: 2,  # "Apple" ve třetí větě je na pozici 2
}

# Extrakce embeddings pro tato slova
target_words_embeddings = [embeddings[i, idx, :] for i, idx in word_indices.items()]

# PCA pro vizualizaci
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(np.array(target_words_embeddings))

# Vykreslení sémantického prostoru
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='steelblue')

# Přidání popisků
for i, (sent_idx, word_idx) in enumerate(word_indices.items()):
    words = sentences[sent_idx].split()
    word = words[word_idx]
    context = sentences[sent_idx]
    plt.annotate(f"{word} ('{context}')", 
                 (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                 fontsize=9)

plt.title("ELMo embeddings pro stejná/podobná slova v různých kontextech")
plt.grid(True)
plt.tight_layout()
plt.show()
```

### BERT Embeddings

BERT (Bidirectional Encoder Representations from Transformers) je kontextový jazykový model založený na architektuře Transformer, který poskytuje hluboké a vysoce kontextově závislé embeddings.

```python
import torch
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Načtení předtrénovaného BERT modelu a tokenizeru
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Funkce pro získání BERT embeddings
def get_bert_embeddings(texts):
    # Tokenizace vstupních textů
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Získání BERT embeddings
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    
    # Poslední skrytý stav [batch_size, sequence_length, hidden_size]
    last_hidden_states = outputs.last_hidden_state
    
    return encoded_inputs, last_hidden_states

# Příklad vět pro ukázku kontextových embeddings
sentences = [
    "The bank is by the river.",
    "I need to bank some money.",
    "He works at the bank.",
    "The river bank was muddy."
]

# Získání BERT embeddings
encoded_inputs, embeddings = get_bert_embeddings(sentences)

# Ukázka rozdílných embeddings pro slovo "bank" v různých kontextech
# Najděme tokeny "bank" v každé větě
bank_token_indices = []

for i, sent in enumerate(sentences):
    # Získání ID tokenů
    tokens = tokenizer.tokenize(sent)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Najdi index tokenu "bank"
    for j, token in enumerate(tokens):
        if token == "bank":
            # +1 kvůli [CLS] tokenu na začátku
            bank_token_indices.append((i, j+1))
            break

# Extrakce embeddings pro "bank" v každé větě
bank_embeddings = [embeddings[i, j, :].numpy() for i, j in bank_token_indices]

# PCA pro vizualizaci
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(np.array(bank_embeddings))

# Vykreslení sémantického prostoru
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='steelblue')

# Přidání popisků
for i, (sent_idx, _) in enumerate(bank_token_indices):
    context = sentences[sent_idx]
    plt.annotate(f"bank v '{context}'", 
                 (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                 fontsize=9)

plt.title("BERT embeddings pro slovo 'bank' v různých kontextech")
plt.grid(True)
plt.tight_layout()
plt.show()
```

## Jazykové modely

Jazykové modely (language models) jsou algoritmy, které se učí pravděpodobnostní rozdělení sekvencí slov. Jejich hlavním úkolem je predikovat pravděpodobnost výskytu sekvence slov nebo následujícího slova v sekvenci.

### Typy jazykových modelů:

1. **Statistické jazykové modely**
   - n-gram modely
   - Hidden Markov Models

2. **Neuronové jazykové modely**
   - RNN/LSTM/GRU jazykové modely
   - Transformer-based modely (BERT, GPT, T5)

## n-gram jazykové modely

n-gramy jsou sekvence n po sobě jdoucích jednotek (typicky slov nebo znaků) v textu. n-gram jazykový model předpovídá další slovo na základě předchozích n-1 slov.

```python
import nltk
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import random

# Stažení potřebných dat
nltk.download('punkt')
nltk.download('brown')
from nltk.corpus import brown

# Příprava trénovacích dat
sentences = brown.sents(categories='news')[:1000]
print(f"Počet trénovacích vět: {len(sentences)}")

# Vytvoření n-gramů a příprava dat pro jazykový model
n = 3  # Budeme používat trigramy
train_data, padded_sents = padded_everygram_pipeline(n, sentences)

# Vytvoření a trénování n-gram modelu s Maximum Likelihood Estimation
model = MLE(n)  # n je velikost n-gramů
model.fit(train_data, padded_sents)

# Funkce pro generování textu pomocí n-gram modelu
def generate_text(model, num_words=20, text_seed=None):
    if text_seed is None:
        # Náhodný začátek
        text = ['<s>']
    else:
        # Použití zadaného seed textu
        text = ['<s>'] + text_seed
    
    # Generování dalších slov
    for _ in range(num_words):
        # Získání posledních (n-1) slov pro kontext
        context = text[-(model.order-1):]
        
        # Predikce dalšího slova
        next_word = model.generate(1, context=context)
        
        # Přidání slova do výsledného textu
        text.append(next_word)
        
        # Kontrola konce věty
        if next_word == '</s>':
            break
    
    # Odstranění značek začátku a konce věty
    return ' '.join(word for word in text if word not in ['<s>', '</s>'])

# Příklad generování textu
generated_text = generate_text(model, num_words=15)
print("Vygenerovaný text:")
print(generated_text)

# Výpočet pravděpodobnosti sekvence slov
def sequence_probability(model, text):
    # Příprava textu
    words = text.split()
    
    # Výpočet pravděpodobnosti
    p = 1.0
    context = ['<s>']
    
    for word in words:
        p *= model.score(word, context)
        
        # Aktualizace kontextu
        context.append(word)
        if len(context) > model.order-1:
            context = context[-(model.order-1):]
    
    return p

# Příklady výpočtu pravděpodobnosti
test_sentences = [
    "the president said",
    "president the said",
    "said president the"
]

for sent in test_sentences:
    prob = sequence_probability(model, sent)
    print(f'P("{sent}") = {prob:.10f}')
```

## Neuronové jazykové modely

Neuronové jazykové modely používají neuronové sítě k modelování pravděpodobnosti sekvencí slov. Na rozdíl od n-gram modelů mohou zachytit dlouhodobé závislosti a sémantické vztahy mezi slovy.

### RNN/LSTM Jazykový Model

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Příprava dat (jednoduchý příklad)
sentences = [
    "Umělá inteligence mění způsob jakým pracujeme",
    "Strojové učení je podoblast umělé inteligence",
    "Deep learning využívá hluboké neuronové sítě",
    "Neuronové sítě se učí z trénovacích dat",
    "Rekurentní neuronové sítě jsou vhodné pro zpracování sekvencí",
    "Konvoluční neuronové sítě se často používají pro analýzu obrazu",
    "Jazykové modely předpovídají pravděpodobnost výskytu sekvence slov",
    "Word embeddings zachycují sémantické vztahy mezi slovy",
    "Transfer learning využívá předtrénované modely pro nové úlohy",
    "Hyperparametry modelu ovlivňují proces učení"
]

# Tokenizace textu
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1
print(f"Celkový počet unikátních slov: {total_words}")

# Vytvoření vstupních sekvencí a cílových slov
sequences = []
for sentence in sentences:
    # Převod věty na sekvenci tokenů
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    
    # Vytvoření vstupních sekvencí a cílových slov
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i]
        sequences.append(n_gram_sequence)

# Padding sekvencí na jednotnou délku
max_sequence_len = max([len(seq) for seq in sequences])
input_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

# Vytvoření vstupů a výstupů pro model
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot kódování výstupů
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Vytvoření modelu
embedding_dim = 100
model = Sequential([
    Embedding(total_words, embedding_dim, input_length=max_sequence_len-1),
    LSTM(150, return_sequences=False),
    Dropout(0.3),
    Dense(total_words, activation='softmax')
])

# Kompilace modelu
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Trénování modelu
early_stopping = EarlyStopping(monitor='loss', patience=10)
history = model.fit(
    X, y,
    epochs=100,
    verbose=1,
    callbacks=[early_stopping]
)

# Vizualizace průběhu trénování
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Přesnost modelu')
plt.xlabel('Epocha')
plt.ylabel('Přesnost')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Ztráta modelu')
plt.xlabel('Epocha')
plt.ylabel('Ztráta')
plt.grid(True)

plt.tight_layout()
plt.show()

# Funkce pro generování textu pomocí natrénovaného modelu
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    result = seed_text
    
    for _ in range(next_words):
        # Tokenizace vstupního textu
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Padding sekvence
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predikce dalšího slova
        predicted = np.argmax(model.predict(token_list), axis=-1)
        
        # Převod tokenu zpět na slovo
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        # Přidání předpovězeného slova k výsledku
        seed_text += " " + output_word
        result += " " + output_word
    
    return result

# Příklad generování textu
seed_text = "Umělá inteligence"
generated_text = generate_text(seed_text, 5, model, tokenizer, max_sequence_len)
print(f"Vygenerovaný text: {generated_text}")
```

## Transformery a pokročilé jazykové modely

Transformery představují architekturu, která způsobila revoluci v NLP. Využívají mechanismus pozornosti (attention mechanism) místo rekurence, což umožňuje efektivnější paralelní zpracování a zachycení dlouhodobých závislostí.

### GPT (Generative Pre-trained Transformer)

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Načtení předtrénovaného GPT-2 modelu a tokenizeru
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Funkce pro generování textu
def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95, num_return=1):
    # Tokenizace promptu
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Generování textu
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=num_return,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Dekódování výstupu
    generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return generated_text

# Příklady generování textu s různými parametry
prompt = "Artificial intelligence is transforming"

print("Standardní generování:")
generated_text = generate_text(model, tokenizer, prompt)
print(generated_text[0])

print("\nGenerování s vyšší teplotou (více kreativní):")
generated_text = generate_text(model, tokenizer, prompt, temperature=1.5)
print(generated_text[0])

print("\nGenerování s nižší teplotou (více deterministické):")
generated_text = generate_text(model, tokenizer, prompt, temperature=0.7)
print(generated_text[0])

# Příklad zero-shot generování s koncem věty
prompt = "Translate the following English text to French: 'Hello, how are you?'"
generated_text = generate_text(model, tokenizer, prompt)
print("\nZero-shot překlad:")
print(generated_text[0])
```

### BERT (Bidirectional Encoder Representations from Transformers)

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM
import matplotlib.pyplot as plt
import numpy as np

# Načtení předtrénovaného BERT modelu a tokenizeru
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Funkce pro predikci maskovaného slova
def predict_masked_word(text, model, tokenizer, top_k=5):
    # Tokenizace textu s maskovaným tokenem
    inputs = tokenizer(text, return_tensors='pt')
    
    # Najití pozice maskovaného tokenu
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    
    # Výpočet predikce
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Získání pravděpodobností pro všechny tokeny na pozici masky
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    
    # Získání top k tokenů s nejvyšší pravděpodobností
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices.tolist()[0]
    
    # Převod tokenů zpět na slova a získání pravděpodobností
    probs = torch.nn.functional.softmax(mask_token_logits, dim=-1)
    top_k_probs = [probs[0, token].item() for token in top_k_tokens]
    top_k_tokens = [tokenizer.convert_ids_to_tokens([token])[0] for token in top_k_tokens]
    
    return list(zip(top_k_tokens, top_k_probs))

# Příklady predikce maskovaných slov
masked_texts = [
    "The [MASK] sat on the mat.",
    "I like to [MASK] in my free time.",
    "Paris is the capital of [MASK].",
    "The [MASK] went to the store to buy some groceries.",
    "Deep learning is a branch of [MASK] learning."
]

for text in masked_texts:
    predictions = predict_masked_word(text, model, tokenizer)
    print(f"\nText: {text}")
    print("Predikce:")
    for word, prob in predictions:
        print(f"  {word}: {prob:.6f}")

# Vizualizace predikcí
def visualize_predictions(text, predictions):
    words = [word for word, _ in predictions]
    probs = [prob for _, prob in predictions]
    
    plt.figure(figsize=(10, 5))
    plt.bar(words, probs, color='steelblue')
    plt.xlabel("Predikované slovo")
    plt.ylabel("Pravděpodobnost")
    plt.title(f"Top 5 predikcí pro: '{text}'")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Vizualizace pro vybraný příklad
visualize_predictions(masked_texts[4], predict_masked_word(masked_texts[4], model, tokenizer))
```

## Vyhodnocování jazykových modelů

Jazykové modely jsou typicky hodnoceny pomocí metriky zvané perplexita, která je definována jako exponenciál průměrné negativní log-likelihood.

```python
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForMaskedLM

# Načtení potřebných modelů
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Výpočet perplexity pro GPT-2 model
def calculate_perplexity(text, model, tokenizer):
    # Tokenizace textu
    encodings = tokenizer(text, return_tensors='pt')
    
    # Získání ID tokenů
    input_ids = encodings.input_ids
    
    # Výpočet cross entropy loss, který použijeme pro výpočet perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    # Perplexity je vypočtena jako e^(loss)
    perplexity = torch.exp(loss)
    
    return perplexity.item()

# Příklady textů pro vyhodnocení
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "I enjoy listening to music while I work.",
    "Artificial intelligence has made significant advances in recent years.",
    "The capital of France is Paris.",
    "qwerty uiop asdf ghjkl zxcvbnm.",  # Náhodný text
]

# Výpočet perplexity pro každý text
print("Perplexity GPT-2 modelu pro různé texty:")
for text in test_texts:
    perplexity = calculate_perplexity(text, gpt2_model, gpt2_tokenizer)
    print(f"  '{text}': {perplexity:.2f}")

# Vizualizace perplexity
def visualize_perplexity(texts, perplexities):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(texts)), perplexities, color='steelblue')
    plt.xlabel("Text")
    plt.ylabel("Perplexity")
    plt.title("Perplexity GPT-2 modelu pro různé texty")
    plt.xticks(range(len(texts)), [f"Text {i+1}" for i in range(len(texts))])
    
    # Přidání hodnot nad sloupce
    for i, v in enumerate(perplexities):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.show()

# Výpočet a vizualizace perplexity
perplexities = [calculate_perplexity(text, gpt2_model, gpt2_tokenizer) for text in test_texts]
visualize_perplexity(test_texts, perplexities)
```

## Praktické aplikace

Word embeddings a jazykové modely mají širokou škálu aplikací v oblasti zpracování přirozeného jazyka:

1. **Sentiment analýza** - klasifikace textu podle nálady nebo názoru
2. **Rozpoznávání pojmenovaných entit** - identifikace osob, míst a organizací v textu
3. **Strojový překlad** - překládání z jednoho jazyka do druhého
4. **Shrnutí textu** - vytváření krátkých souhrnů dlouhých dokumentů
5. **Chatboty a konverzační agenti** - systémy pro automatickou konverzaci
6. **Generování textu** - automatické vytváření článků, popisů, poezie atd.
7. **Doplňování textu** - predikce a doplňování nedokončených vět

### Příklad sentiment analýzy s word embeddings

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Generace syntetických dat pro sentiment analýzu
def generate_sentiment_data(n_samples=1000):
    positive_phrases = [
        "Miluji tento film", "Skvělý výkon", "Úžasná atmosféra", "Perfektní zážitek",
        "Velmi doporučuji", "Nejlepší kniha", "Báječná restaurace", "Vynikající služby",
        "Naprosto spokojený", "Fenomenální produkt"
    ]
    
    negative_phrases = [
        "Hrozný film", "Špatný výkon", "Nepříjemná atmosféra", "Hrozný zážitek",
        "Nedoporučuji", "Nejhorší kniha", "Špatná restaurace", "Hrozné služby",
        "Naprosto nespokojený", "Hrozný produkt"
    ]
    
    neutral_phrases = [
        "Průměrný film", "Standardní výkon", "Běžná atmosféra", "Normální zážitek",
        "Možná doporučím", "Běžná kniha", "Průměrná restaurace", "Standardní služby",
        "Částečně spokojený", "Průměrný produkt"
    ]
    
    # Generování dat s různými kombinacemi frází
    data = []
    
    for _ in range(n_samples // 3):
        # Pozitivní recenze
        pos_text = np.random.choice(positive_phrases) + " " + np.random.choice(positive_phrases)
        if np.random.rand() > 0.8:  # Občas přidat neutrální frázi
            pos_text += " " + np.random.choice(neutral_phrases)
        data.append((pos_text, "positive"))
        
        # Negativní recenze
        neg_text = np.random.choice(negative_phrases) + " " + np.random.choice(negative_phrases)
        if np.random.rand() > 0.8:  # Občas přidat neutrální frázi
            neg_text += " " + np.random.choice(neutral_phrases)
        data.append((neg_text, "negative"))
        
        # Neutrální recenze
        neu_text = np.random.choice(neutral_phrases) + " " + np.random.choice(neutral_phrases)
        if np.random.rand() > 0.5:  # Občas přidat pozitivní nebo negativní frázi
            if np.random.rand() > 0.5:
                neu_text += " " + np.random.choice(positive_phrases)
            else:
                neu_text += " " + np.random.choice(negative_phrases)
        data.append((neu_text, "neutral"))
    
    # Vytvoření DataFrame
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    
    return df

# Generování dat
sentiment_data = generate_sentiment_data(1200)
print(sentiment_data.head())
print(sentiment_data['sentiment'].value_counts())

# Rozdělení na trénovací a testovací sadu
X_train, X_test, y_train, y_test = train_test_split(
    sentiment_data['text'], 
    sentiment_data['sentiment'], 
    test_size=0.2, 
    random_state=42
)

# Tokenizace textu
max_features = 5000  # Maximum počet slov ve slovníku
maxlen = 100  # Maximum počet slov v recenzi

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# Kódování cílových proměnných
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# Převod na kategoriální proměnné
y_train_cat = tf.keras.utils.to_categorical(y_train_enc, num_classes=3)
y_test_cat = tf.keras.utils.to_categorical(y_test_enc, num_classes=3)

# Vytvoření modelu s vlastní embedding vrstvou
embedding_dim = 100  # Dimenze embedding vektoru
vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Trénování modelu
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(
    X_train_pad, y_train_cat,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Vizualizace průběhu trénování
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Vyhodnocení modelu
loss, accuracy = model.evaluate(X_test_pad, y_test_cat)
print(f"Test accuracy: {accuracy:.4f}")

# Predikce na nových datech
new_texts = [
    "Ten film byl naprosto úžasný, miluji ho!",
    "Nemůžu uvěřit, jak špatný ten film byl, nedoporučuji.",
    "Film byl celkem dobrý, ale nic extra."
]

# Tokenizace a padding nových textů
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=maxlen)

# Predikce
predictions = model.predict(new_padded)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Výpis výsledků
print("\nPredikce sentimentu pro nové texty:")
for i, text in enumerate(new_texts):
    print(f"Text: {text}")
    print(f"Predikovaný sentiment: {predicted_labels[i]}")
    print(f"Konfidence: {predictions[i][predicted_classes[i]]:.4f}")
    print()
```

## Shrnutí

Word embeddings a jazykové modely představují základní stavební kameny pro moderní systémy zpracování přirozeného jazyka. Word embeddings umožňují reprezentovat slova jako vektory v sémantickém prostoru, kde podobná slova mají podobné vektory, což umožňuje zachytit významové a syntaktické vztahy mezi slovy.

Existuje několik typů word embeddings, od statických reprezentací jako Word2Vec, GloVe a FastText, až po kontextové embeddings jako ELMo a BERT, které generují různé vektorové reprezentace stejného slova v závislosti na jeho kontextu.

Jazykové modely se používají k předpovídání pravděpodobnosti výskytu sekvence slov. Od jednoduchých n-gram modelů, přes rekurentní neuronové sítě, až po moderní architektury založené na Transformerech, jako jsou GPT a BERT. Tyto modely nacházejí široké uplatnění v různých aplikacích NLP, včetně strojového překladu, generování textu, sentiment analýzy a mnoha dalších.

S příchodem velkých jazykových modelů (LLM) založených na architektuře Transformer došlo k výraznému pokroku ve schopnostech NLP systémů, které nyní dokáží generovat koherentní text, odpovídat na otázky, shrnovat dokumenty a provádět mnohé další úkoly s impresivní kvalitou.