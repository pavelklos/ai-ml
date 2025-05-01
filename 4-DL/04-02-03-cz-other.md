# GRU a obousměrné RNN sítě

## Obsah
1. Úvod do GRU sítí
2. Architektura GRU
3. Porovnání GRU a LSTM
4. Implementace GRU v TensorFlow/Keras
5. Implementace GRU v PyTorch
6. Úvod do obousměrných RNN
7. Architektura obousměrných RNN
8. Implementace obousměrných RNN v TensorFlow/Keras
9. Implementace obousměrných RNN v PyTorch
10. Praktické aplikace a příklady
11. Shrnutí a nejlepší postupy

## Úvod do GRU sítí

GRU (Gated Recurrent Unit) je typ rekurentní neuronové sítě, který byl představen v roce 2014 jako alternativa k LSTM. GRU je navržena ke stejnému účelu jako LSTM - řešení problému mizejícího gradientu v tradičních RNN. GRU dosahuje podobných výsledků jako LSTM, ale s jednodušší architekturou, která vede k nižšímu počtu parametrů a potenciálně rychlejšímu trénování.

**Klíčové vlastnosti GRU:**
- Jednodušší architektura než LSTM (2 brány místo 3)
- Nižší výpočetní náročnost
- Podobný výkon při zachycování dlouhodobých závislostí
- Méně parametrů k naučení
- Neobsahuje samostatný buněčný stav jako LSTM

## Architektura GRU

GRU obsahuje dvě hlavní brány:
- **Update gate (aktualizační brána)** - Rozhoduje, kolik informací z předchozího skrytého stavu by mělo být zachováno
- **Reset gate (resetovací brána)** - Určuje, jak kombinovat nový vstup s předchozím skrytým stavem

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Schematické znázornění výpočtu v GRU
def gru_computation_flow(x_t, h_prev):
    """
    Ilustrativní funkce znázorňující tok výpočtů v GRU buňce
    """
    # Parametry GRU
    W_z = np.random.randn(3, 3)  # váhy pro update gate
    U_z = np.random.randn(3, 3)  # rekurentní váhy pro update gate
    b_z = np.zeros(3)  # bias pro update gate
    
    W_r = np.random.randn(3, 3)  # váhy pro reset gate
    U_r = np.random.randn(3, 3)  # rekurentní váhy pro reset gate
    b_r = np.zeros(3)  # bias pro reset gate
    
    W_h = np.random.randn(3, 3)  # váhy pro kandidátní aktivaci
    U_h = np.random.randn(3, 3)  # rekurentní váhy pro kandidátní aktivaci
    b_h = np.zeros(3)  # bias pro kandidátní aktivaci
    
    # GRU výpočty
    # Update gate
    z_t = sigmoid(np.dot(W_z, x_t) + np.dot(U_z, h_prev) + b_z)
    
    # Reset gate
    r_t = sigmoid(np.dot(W_r, x_t) + np.dot(U_r, h_prev) + b_r)
    
    # Kandidátní aktivace
    h_tilde = np.tanh(np.dot(W_h, x_t) + np.dot(U_h, r_t * h_prev) + b_h)
    
    # Nový skrytý stav
    h_t = (1 - z_t) * h_prev + z_t * h_tilde
    
    return h_t, z_t, r_t, h_tilde

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Ukázka GRU výpočtu s náhodnými daty
x_t = np.random.randn(3, 1)  # vstupní vektor
h_prev = np.random.randn(3, 1)  # předchozí skrytý stav

h_t, z_t, r_t, h_tilde = gru_computation_flow(x_t, h_prev)
```

Matematicky lze GRU vyjádřit následovně:

```
z_t = σ(W_z * x_t + U_z * h_{t-1} + b_z)  # update gate
r_t = σ(W_r * x_t + U_r * h_{t-1} + b_r)  # reset gate
h_tilde = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h)  # kandidátní aktivace
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde  # nový skrytý stav
```

Kde:
- σ je sigmoid funkce
- ⊙ reprezentuje Hadamardův součin (násobení prvek po prvku)
- W, U jsou váhové matice
- b jsou bias vektory
- x_t je vstupní vektor v čase t
- h_t je skrytý stav v čase t

## Porovnání GRU a LSTM

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense

# Generování jednoduchých dat pro demonstraci
def generate_sequence_data(n_samples=1000, seq_length=20, delay=5):
    """
    Generuje syntetická data pro porovnání GRU a LSTM
    """
    X = np.zeros((n_samples, seq_length, 1))
    Y = np.zeros((n_samples, 1))
    
    for i in range(n_samples):
        # Náhodná sekvence
        signal = np.random.normal(0, 0.1, seq_length)
        # Přidáme charakteristický vzor na začátek
        pattern = np.sin(np.linspace(0, 4*np.pi, delay))
        signal[:delay] += pattern
        
        X[i, :, 0] = signal
        # Cíl: 1 pokud sequence obsahuje vzor na začátku, jinak 0
        Y[i, 0] = 1
    
    # Mícháme data a přiřadíme polovině dat label 0
    idx = np.random.permutation(n_samples)
    X = X[idx]
    Y = Y[idx]
    Y[n_samples//2:] = 0
    X[n_samples//2:, :delay, 0] = np.random.normal(0, 0.1, (n_samples//2, delay))
    
    return X, Y

# Generování dat
X, Y = generate_sequence_data(n_samples=2000, seq_length=20, delay=5)

# Rozdělení na trénovací a testovací data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# GRU model
gru_model = Sequential([
    GRU(50, input_shape=(X.shape[1], X.shape[2])),
    Dense(1, activation='sigmoid')
])

gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# LSTM model
lstm_model = Sequential([
    LSTM(50, input_shape=(X.shape[1], X.shape[2])),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trénování modelů
gru_history = gru_model.fit(X_train, Y_train, epochs=10, batch_size=32, 
                           validation_data=(X_test, Y_test), verbose=0)
lstm_history = lstm_model.fit(X_train, Y_train, epochs=10, batch_size=32, 
                             validation_data=(X_test, Y_test), verbose=0)

# Porovnání výkonu
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(gru_history.history['accuracy'], label='GRU trénovací přesnost')
plt.plot(gru_history.history['val_accuracy'], label='GRU validační přesnost')
plt.plot(lstm_history.history['accuracy'], label='LSTM trénovací přesnost')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM validační přesnost')
plt.title('GRU vs LSTM: Přesnost')
plt.xlabel('Epocha')
plt.ylabel('Přesnost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(gru_history.history['loss'], label='GRU trénovací loss')
plt.plot(gru_history.history['val_loss'], label='GRU validační loss')
plt.plot(lstm_history.history['loss'], label='LSTM trénovací loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM validační loss')
plt.title('GRU vs LSTM: Loss')
plt.xlabel('Epocha')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

### Klíčové rozdíly mezi GRU a LSTM:
1. **Architektura**: GRU má 2 brány (update, reset), zatímco LSTM má 3 brány (input, forget, output)
2. **Paměťový mechanismus**: LSTM má oddělený buněčný stav, GRU kombinuje řízení aktualizace a zapomenutí do jedné brány
3. **Počet parametrů**: GRU má méně parametrů než LSTM
4. **Výpočetní náročnost**: GRU je výpočetně méně náročná než LSTM
5. **Výkon**: Výkon obou architektur je podobný, s mírnou výhodou pro jednu či druhou v závislosti na konkrétní úloze

## Implementace GRU v TensorFlow/Keras

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Příprava dat - jednoduchá klasifikační úloha na časových řadách
def create_time_series_data(n_samples=1000, sequence_length=50, n_features=1, n_classes=3):
    """
    Vytváří syntetická data s různými vzory pro klasifikaci
    """
    X = np.zeros((n_samples, sequence_length, n_features))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        pattern_type = i % n_classes
        if pattern_type == 0:  # Rostoucí trend
            X[i, :, 0] = np.linspace(0, 1, sequence_length) + 0.1 * np.random.randn(sequence_length)
            y[i] = 0
        elif pattern_type == 1:  # Klesající trend
            X[i, :, 0] = np.linspace(1, 0, sequence_length) + 0.1 * np.random.randn(sequence_length)
            y[i] = 1
        else:  # Sinusový vzor
            X[i, :, 0] = 0.5 * np.sin(np.linspace(0, 4*np.pi, sequence_length)) + 0.1 * np.random.randn(sequence_length)
            y[i] = 2
    
    # One-hot kódování třídy
    return X, to_categorical(y, num_classes=n_classes)

# Vytvoření dat
n_classes = 3
X, y = create_time_series_data(n_samples=1500, sequence_length=50, n_features=1, n_classes=n_classes)

# Rozdělení na trénovací a testovací data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vytvoření modelu s GRU vrstvou
model = Sequential([
    # První GRU vrstva, return_sequences=True znamená, že vracíme výstupy pro všechny časové kroky
    GRU(64, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),  # Dropout pro prevenci přetrénování
    
    # Druhá GRU vrstva
    GRU(32, activation='tanh'),
    Dropout(0.2),
    
    # Výstupní vrstva
    Dense(n_classes, activation='softmax')
])

# Kompilace modelu
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Zobrazení struktury modelu
model.summary()

# Trénování modelu
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Vyhodnocení modelu
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Vizualizace průběhu trénování
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trénovací přesnost')
plt.plot(history.history['val_accuracy'], label='Validační přesnost')
plt.title('Přesnost modelu')
plt.xlabel('Epocha')
plt.ylabel('Přesnost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trénovací loss')
plt.plot(history.history['val_loss'], label='Validační loss')
plt.title('Loss modelu')
plt.xlabel('Epocha')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

## Implementace GRU v PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Definice GRU modelu v PyTorch
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU vrstvy
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout vrstva
        self.dropout = nn.Dropout(dropout_rate)
        
        # Plně propojená výstupní vrstva
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Inicializace počátečního skrytého stavu
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass přes GRU
        out, _ = self.gru(x, h0)
        
        # Použijeme pouze výstup posledního časového kroku
        out = out[:, -1, :]
        
        # Dropout
        out = self.dropout(out)
        
        # Plně propojená vrstva
        out = self.fc(out)
        
        return out

# Příprava dat pro PyTorch (použijeme stejná data jako v TF příkladu)
def create_time_series_data(n_samples=1000, sequence_length=50, n_features=1, n_classes=3):
    X = np.zeros((n_samples, sequence_length, n_features))
    y = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        pattern_type = i % n_classes
        if pattern_type == 0:  # Rostoucí trend
            X[i, :, 0] = np.linspace(0, 1, sequence_length) + 0.1 * np.random.randn(sequence_length)
            y[i] = 0
        elif pattern_type == 1:  # Klesající trend
            X[i, :, 0] = np.linspace(1, 0, sequence_length) + 0.1 * np.random.randn(sequence_length)
            y[i] = 1
        else:  # Sinusový vzor
            X[i, :, 0] = 0.5 * np.sin(np.linspace(0, 4*np.pi, sequence_length)) + 0.1 * np.random.randn(sequence_length)
            y[i] = 2
    
    return X, y

# Vytvoření dat
X, y = create_time_series_data(n_samples=1500, sequence_length=50, n_features=1, n_classes=3)

# Rozdělení na trénovací a testovací data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konverze na PyTorch tenzory
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Hyperparametry
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 3
dropout_rate = 0.2
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# Inicializace modelu, loss funkce a optimizeru
model = GRUModel(input_size, hidden_size, num_layers, output_size, dropout_rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trénování modelu
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Trénování po batchi
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass a optimalizace
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Výpočet přesnosti
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    train_loss = running_loss / (len(X_train) / batch_size)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluace na testovacích datech
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
        
        _, predicted = torch.max(test_outputs.data, 1)
        test_acc = (predicted == y_test).sum().item() / len(y_test)
        test_accs.append(test_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# Vizualizace výsledků
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Trénovací loss')
plt.plot(test_losses, label='Validační loss')
plt.title('Loss modelu')
plt.xlabel('Epocha')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Trénovací přesnost')
plt.plot(test_accs, label='Validační přesnost')
plt.title('Přesnost modelu')
plt.xlabel('Epocha')
plt.ylabel('Přesnost')
plt.legend()

plt.tight_layout()
plt.show()
```

## Úvod do obousměrných RNN

Obousměrné rekurentní neuronové sítě (Bidirectional RNNs, BiRNNs) jsou rozšířením standardních RNN, které zpracovávají vstupní sekvenci v obou směrech - dopředu i dozadu. Toto umožňuje modelu zachytit kontext z obou směrů sekvence, což je zvláště užitečné u úloh jako je zpracování přirozeného jazyka, kde je důležitý kontext jak z předcházejících, tak z následujících slov.

**Klíčové vlastnosti BiRNN:**
- Zpracování sekvence v obou směrech (dopředu i zpět)
- Zachycení komplexnějšího kontextu
- Vhodné pro úlohy, kde je důležitá informace z celé sekvence
- Mohou být kombinovány s jakýmkoliv typem RNN buňky (obyčejná RNN, LSTM, GRU)

## Architektura obousměrných RNN

Obousměrná RNN se skládá ze dvou samostatných rekurentních sítí:
1. **Dopředná RNN** - zpracovává sekvenci od začátku do konce
2. **Zpětná RNN** - zpracovává sekvenci od konce k začátku

Výstupy obou sítí jsou pak typicky zkombinovány (např. konkatenací nebo sčítáním) a použity pro predikci.

```python
import numpy as np

# Schematické znázornění architektury BiRNN
def bidirectional_rnn_schema(input_sequence, forward_weights, backward_weights):
    """
    Ilustrativní funkce znázorňující tok dat v BiRNN
    """
    sequence_length = len(input_sequence)
    hidden_size = forward_weights.shape[0]
    
    # Inicializace skrytých stavů
    forward_states = np.zeros((sequence_length, hidden_size))
    backward_states = np.zeros((sequence_length, hidden_size))
    combined_states = np.zeros((sequence_length, 2 * hidden_size))
    
    # Dopředný průchod - od začátku do konce
    h_forward = np.zeros(hidden_size)
    for t in range(sequence_length):
        h_forward = np.tanh(np.dot(forward_weights, input_sequence[t]) + h_forward)
        forward_states[t] = h_forward
    
    # Zpětný průchod - od konce k začátku
    h_backward = np.zeros(hidden_size)
    for t in range(sequence_length - 1, -1, -1):
        h_backward = np.tanh(np.dot(backward_weights, input_sequence[t]) + h_backward)
        backward_states[t] = h_backward
    
    # Kombinace výstupů
    for t in range(sequence_length):
        combined_states[t] = np.concatenate([forward_states[t], backward_states[t]])
    
    return forward_states, backward_states, combined_states

# Ukázkový příklad s náhodnými daty
input_sequence = np.random.randn(10, 5)  # sekvence 10 vektorů o velikosti 5
forward_weights = np.random.randn(3, 5)  # váhy pro dopředný průchod, skrytý stav velikosti 3
backward_weights = np.random.randn(3, 5)  # váhy pro zpětný průchod, skrytý stav velikosti 3

forward_states, backward_states, combined_states = bidirectional_rnn_schema(
    input_sequence, forward_weights, backward_weights)
```

## Implementace obousměrných RNN v TensorFlow/Keras

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Příklad s textovou klasifikací - sentiment analýza

# Vytvoření syntetických dat
texts = [
    "Tento film se mi velmi líbil, doporučuji všem.",
    "Skvělý výkon herců, úžasné efekty.",
    "Nejhorší film, který jsem kdy viděl.",
    "Ztráta času a peněz, absolutně nedoporučuji.",
    "Film byl průměrný, nic výjimečného.",
    "Herci odvedli dobrou práci, ale scénář byl slabý.",
    "Fascinující příběh s nečekaným koncem.",
    "Nezajímavé a nudné, usnul jsem v polovině.",
    "Perfektní mix humoru a dramatu.",
    "Úplná katastrofa, nemá to žádný smysl."
]

# Vytvoření labelů - 1 pro pozitivní, 0 pro negativní
labels = np.array([1, 1, 0, 0, 0.5, 0.5, 1, 0, 1, 0])

# Příprava textu - tokenizace
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sekvencí na stejnou délku
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Rozdělení na trénovací a testovací data
indices = np.arange(len(padded_sequences))
np.random.shuffle(indices)
padded_sequences = padded_sequences[indices]
labels = labels[indices]

train_size = int(0.8 * len(padded_sequences))
X_train = padded_sequences[:train_size]
y_train = labels[:train_size]
X_test = padded_sequences[train_size:]
y_test = labels[train_size:]

# Vytvoření modelu s obousměrným GRU
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16
model = Sequential([
    # Embedding vrstva
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    
    # Obousměrný GRU
    Bidirectional(GRU(32, return_sequences=True)),
    
    # Další obousměrný GRU
    Bidirectional(GRU(16)),
    
    # Výstupní vrstva
    Dense(1, activation='sigmoid')
])

# Kompilace modelu
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

model.summary()

# Trénování modelu
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=2,
    verbose=1
)

# Vyhodnocení modelu
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Vizualizace průběhu trénování
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trénovací přesnost')
plt.plot(history.history['val_accuracy'], label='Validační přesnost')
plt.title('Přesnost modelu')
plt.xlabel('Epocha')
plt.ylabel('Přesnost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trénovací loss')
plt.plot(history.history['val_loss'], label='Validační loss')
plt.title('Loss modelu')
plt.xlabel('Epocha')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

## Implementace obousměrných RNN v PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Definice modelu obousměrného GRU v PyTorch
class BidirectionalGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(BidirectionalGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Obousměrný GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Klíčový parametr pro obousměrný GRU
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Plně propojená vrstva (všimněte si *2 pro hidden_size - kvůli obousměrnosti)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # Inicializace počátečního skrytého stavu
        # *2 pro počet směrů (dopředný a zpětný)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass přes GRU
        out, _ = self.gru(x, h0)
        
        # Použijeme pouze výstup posledního časového kroku
        out = out[:, -1, :]
        
        # Dropout
        out = self.dropout(out)
        
        # Plně propojená vrstva
        out = self.fc(out)
        
        return out

# Příprava dat pro PyTorch - stejná funkce jako v předchozích příkladech
def create_time_series_data(n_samples=1000, sequence_length=50, n_features=1, n_classes=3):
    X = np.zeros((n_samples, sequence_length, n_features))
    y = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        pattern_type = i % n_classes
        if pattern_type == 0:
            X[i, :, 0] = np.linspace(0, 1, sequence_length) + 0.1 * np.random.randn(sequence_length)
            y[i] = 0
        elif pattern_type == 1:
            X[i, :, 0] = np.linspace(1, 0, sequence_length) + 0.1 * np.random.randn(sequence_length)
            y[i] = 1
        else:
            X[i, :, 0] = 0.5 * np.sin(np.linspace(0, 4*np.pi, sequence_length)) + 0.1 * np.random.randn(sequence_length)
            y[i] = 2
    
    return X, y

# Vytvoření dat
X, y = create_time_series_data(n_samples=1500, sequence_length=50, n_features=1, n_classes=3)

# Rozdělení na trénovací a testovací data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konverze na PyTorch tenzory
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Hyperparametry
input_size = 1
hidden_size = 32
num_layers = 2
output_size = 3
dropout_rate = 0.2
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# Inicializace modelu, loss funkce a optimizeru
model = BidirectionalGRUModel(input_size, hidden_size, num_layers, output_size, dropout_rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trénování modelu
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Trénování po batchi
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass a optimalizace
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Výpočet přesnosti
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    train_loss = running_loss / (len(X_train) / batch_size)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluace na testovacích datech
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
        
        _, predicted = torch.max(test_outputs.data, 1)
        test_acc = (predicted == y_test).sum().item() / len(y_test)
        test_accs.append(test_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# Vizualizace výsledků
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Trénovací loss')
plt.plot(test_losses, label='Validační loss')
plt.title('Loss modelu')
plt.xlabel('Epocha')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Trénovací přesnost')
plt.plot(test_accs, label='Validační přesnost')
plt.title('Přesnost modelu')
plt.xlabel('Epocha')
plt.ylabel('Přesnost')
plt.legend()

plt.tight_layout()
plt.show()
```

## Praktické aplikace a příklady

### Příklad: Predikce časové řady s BiGRU

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Generování syntetické časové řady
def generate_time_series(n_samples=1000):
    time = np.linspace(0, 50, n_samples)
    # Kombinace několika sinusovek a lineárního trendu
    series = 0.5 * np.sin(0.1 * time) + 0.2 * np.sin(0.3 * time) + 0.01 * time
    series += 0.1 * np.random.randn(n_samples)  # Přidání šumu
    return series

# Příprava dat ve formě pohybujícího se okna
def create_dataset(series, look_back=50, forecast_horizon=10):
    X, y = [], []
    for i in range(len(series) - look_back - forecast_horizon):
        feature = series[i:i+look_back]
        target = series[i+look_back:i+look_back+forecast_horizon]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)

# Generování a příprava dat
series = generate_time_series()
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

look_back = 50  # Velikost okna pro vstupy
forecast_horizon = 10  # Počet kroků, které chceme předpovědět
X, y = create_dataset(series_scaled, look_back, forecast_horizon)

# Reshape pro input do RNN: [vzorky, časové_kroky, příznaky]
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], 1)

# Rozdělení na trénovací a testovací sady
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Vytvoření modelu BiGRU pro predikci více kroků
model = Sequential([
    Bidirectional(GRU(64, return_sequences=True), input_shape=(look_back, 1)),
    Dropout(0.2),
    Bidirectional(GRU(32)),
    Dropout(0.2),
    Dense(forecast_horizon)  # Předpověď příštích n kroků
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# Trénování modelu
history = model.fit(
    X_train, y_train.reshape(y_train.shape[0], forecast_horizon),
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test.reshape(y_test.shape[0], forecast_horizon)),
    verbose=1
)

# Predikce
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Reshape predikcí pro inverse transform
train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

# Inverzní transformace pro získání původní škály
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Výpočet MSE
train_mse = mean_squared_error(
    y_train.reshape(-1, 1), 
    scaler.transform(train_predict).reshape(y_train.shape)
)
test_mse = mean_squared_error(
    y_test.reshape(-1, 1), 
    scaler.transform(test_predict).reshape(y_test.shape)
)
print(f"Train MSE: {train_mse:.6f}")
print(f"Test MSE: {test_mse:.6f}")

# Vizualizace průběhu trénování
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Trénovací MSE')
plt.plot(history.history['val_loss'], label='Validační MSE')
plt.title('MSE v průběhu trénování')
plt.xlabel('Epocha')
plt.ylabel('MSE')
plt.legend()

# Vizualizace predikce (pouze část testovacích dat pro přehlednost)
test_idx = 50  # Index vzorku z testovací sady pro vizualizaci
plt.subplot(1, 2, 2)

actual_values = series_scaled[train_size + test_idx + look_back:
                              train_size + test_idx + look_back + forecast_horizon]
predicted_values = scaler.transform(test_predict.reshape(-1, 1)).flatten()[
    test_idx*forecast_horizon:(test_idx+1)*forecast_horizon]

plt.plot(range(forecast_horizon), actual_values, label='Skutečné hodnoty')
plt.plot(range(forecast_horizon), predicted_values, label='Predikované hodnoty')
plt.title('Predikce vs. Skutečnost')
plt.xlabel('Časový krok')
plt.ylabel('Hodnota')
plt.legend()

plt.tight_layout()
plt.show()
```

### Příklad: Klasifikace sekvencí s BiGRU pro rozpoznávání akcí

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generování syntetických dat pro rozpoznávání akcí
def generate_action_data(n_samples=1000, sequence_length=30, n_features=3, n_classes=4):
    X = np.zeros((n_samples, sequence_length, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        action_type = i % n_classes
        
        if action_type == 0:  # "Chůze" - periodický vzor
            freq = 0.5
            t = np.linspace(0, 4*np.pi, sequence_length)
            X[i, :, 0] = np.sin(freq * t) + 0.1 * np.random.randn(sequence_length)
            X[i, :, 1] = np.cos(freq * t) + 0.1 * np.random.randn(sequence_length)
            X[i, :, 2] = 0.2 * np.random.randn(sequence_length)
            
        elif action_type == 1:  # "Běh" - rychlejší periodický vzor
            freq = 1.0
            t = np.linspace(0, 4*np.pi, sequence_length)
            X[i, :, 0] = 1.5 * np.sin(freq * t) + 0.15 * np.random.randn(sequence_length)
            X[i, :, 1] = 1.5 * np.cos(freq * t) + 0.15 * np.random.randn(sequence_length)
            X[i, :, 2] = 0.3 * np.random.randn(sequence_length)
            
        elif action_type == 2:  # "Skok" - jeden výrazný pulz
            X[i, :, 0] = 0.1 * np.random.randn(sequence_length)
            X[i, :, 1] = 0.1 * np.random.randn(sequence_length)
            peak_pos = np.random.randint(5, sequence_length - 5)
            X[i, peak_pos-3:peak_pos+3, 2] = 2.0 + 0.2 * np.random.randn(6)
            
        else:  # "Stání" - téměř konstantní
            X[i, :, 0] = 0.05 * np.random.randn(sequence_length)
            X[i, :, 1] = 0.05 * np.random.randn(sequence_length)
            X[i, :, 2] = 0.05 * np.random.randn(sequence_length)
        
        y[i] = action_type
    
    # One-hot kódování třídy
    return X, to_categorical(y, num_classes=n_classes)

# Vytvoření dat
n_classes = 4
X, y = generate_action_data(n_samples=2000, sequence_length=30, n_features=3, n_classes=n_classes)

# Rozdělení na trénovací a testovací data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vytvoření modelu BiGRU pro klasifikaci sekvencí
model = Sequential([
    # První vrstva BiGRU
    Bidirectional(GRU(64, return_sequences=True), input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    
    # Druhá vrstva BiGRU
    Bidirectional(GRU(32)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Výstupní vrstva
    Dense(n_classes, activation='softmax')
])

# Kompilace modelu
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Trénování modelu
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=1
)

# Vyhodnocení modelu
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Vizualizace průběhu trénování
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trénovací přesnost')
plt.plot(history.history['val_accuracy'], label='Validační přesnost')
plt.title('Přesnost modelu')
plt.xlabel('Epocha')
plt.ylabel('Přesnost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trénovací loss')
plt.plot(history.history['val_loss'], label='Validační loss')
plt.title('Loss modelu')
plt.xlabel('Epocha')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

## Shrnutí a nejlepší postupy

### Kdy použít GRU vs. LSTM vs. BiRNN:

1. **GRU**:
   - Menší datové množiny
   - Potřeba rychlejšího trénování
   - Omezené výpočetní zdroje
   - Úlohy s kratšími nebo středně dlouhými závislostmi

2. **LSTM**:
   - Potřeba zachytit velmi dlouhé závislosti
   - Komplexnější problémy
   - Větší datové množiny
   - Dostatek výpočetních zdrojů

3. **Obousměrné RNN**:
   - Úlohy vyžadující kontext z obou směrů
   - Zpracování přirozeného jazyka
   - Rozpoznávání řeči
   - Analýza časových řad s důležitými vzory v obou směrech

### Nejlepší postupy:

1. **Předzpracování dat**:
   - Normalizace vstupních dat
   - Správné formátování sekvencí
   - Vhodná délka sekvence pro konkrétní problém

2. **Architektura modelu**:
   - Začněte s jednodušším modelem (jednovrstvá GRU/LSTM)
   - Přidávejte složitost postupně
   - Vyzkoušejte jak jednosměrné, tak obousměrné architektury
   - Kombinujte rekurentní vrstvy s dropout a batch normalization pro prevenci přetrénování

3. **Optimalizace**:
   - Gradient clipping pro prevenci explodujícího gradientu
   - Adaptivní optimizery (Adam, RMSprop)
   - Learning rate scheduling
   - Včasné zastavení (early stopping) pro prevenci přetrénování

4. **Kombinace s jinými architekturami**:
   - CNN + RNN pro zpracování obrazových sekvencí
   - Attention mechanismy pro zaměření na důležité části sekvence
   - Transformer sítě pro velmi dlouhé sekvence

### Výhody a nevýhody GRU:

**Výhody**:
- Méně parametrů než LSTM, rychlejší trénování
- Řeší problém mizejícího gradientu
- Podobný výkon jako LSTM v mnoha úlohách
- Jednodušší implementace a ladění

**Nevýhody**:
- Může být méně výkonná než LSTM pro specifické problémy s velmi dlouhými závislostmi
- Chybí oddělený buněčný stav, který poskytuje LSTM

### Výhody a nevýhody obousměrných RNN:

**Výhody**:
- Zachytí kontext z obou směrů sekvence
- Výrazně lepší výkon v NLP úlohách
- Lepší porozumění celkové struktuře sekvence

**Nevýhody**:
- Dvojnásobný počet parametrů oproti jednosměrným sítím
- Vyšší výpočetní náročnost
- Nelze použít pro úlohy vyžadující online inferenci (kdy data přicházejí postupně)
- Potřeba celé sekvence předem

---

## Shrnutí

GRU (Gated Recurrent Unit) sítě představují zjednodušenou alternativu k LSTM sítím, navržené k řešení problému mizejícího gradientu v rekurentních neuronových sítích. Jejich architektura obsahuje dvě hlavní brány - aktualizační a resetovací, což vede k menšímu počtu parametrů a potenciálně rychlejšímu trénování při zachování podobné schopnosti modelovat dlouhodobé závislosti v datech.

Obousměrné RNN (BiRNN) rozšiřují standardní rekurentní sítě tím, že zpracovávají vstupní sekvenci současně v obou směrech - dopředu i dozadu. To umožňuje modelu zachytit kontext z celé sekvence, což je zvláště užitečné pro úlohy jako je zpracování přirozeného jazyka nebo analýza časových řad, kde je důležitý kontext z obou směrů.

Tyto architektury lze implementovat pomocí moderních frameworků jako TensorFlow/Keras nebo PyTorch, které poskytují vysokoúrovňové API pro snadnou práci s těmito modely. GRU a BiRNN nalézají široké využití v oblasti zpracování přirozeného jazyka, predikce časových řad, rozpoznávání řeči a dalších úlohách vyžadujících analýzu sekvenčních dat.

Při volbě konkrétní architektury je třeba zvážit složitost problému, množství dat, výpočetní zdroje a specifické požadavky úlohy. GRU je vhodná pro rychlejší trénování a menší datasety, zatímco obousměrné architektury jsou ideální pro úlohy vyžadující komplexní kontextové informace z celé sekvence.