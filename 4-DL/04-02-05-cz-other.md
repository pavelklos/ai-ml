# Autoencoders a jejich aplikace

## Obsah
1. Úvod do autoenkodérů
2. Základní architektura
3. Typy autoenkodérů
   - Vanilla autoenkodéry
   - Sparse autoenkodéry
   - Denoising autoenkodéry
   - Variační autoenkodéry (VAE)
   - Konvoluční autoenkodéry
4. Implementace v deep learning frameworcích
   - TensorFlow/Keras
   - PyTorch
5. Aplikace autoenkodérů
   - Redukce dimenzionality
   - Detekce anomálií
   - Odstranění šumu z obrázků
   - Generativní modely
   - Doporučovací systémy
6. Pokročilé techniky a variace
   - Conditional VAEs
   - β-VAE
   - VQ-VAE
7. Nejlepší postupy a omezení

## Úvod do autoenkodérů

Autoenkodéry jsou speciálním typem neuronových sítí, které se učí komprimovat a následně rekonstruovat vstupní data. Na rozdíl od klasifikačních nebo regresních sítí, autoenkodéry se učí bez explicitních značek (labels) - jsou to tzv. samoučící se (self-supervised) modely.

Hlavním cílem autoenkodéru je naučit se efektivní vnitřní reprezentaci dat (tzv. latentní prostor), která zachycuje nejdůležitější charakteristiky vstupních dat při nižší dimenzionalitě.

**Klíčové vlastnosti autoenkodérů:**
- Učí se bez nutnosti značených dat
- Umožňují kompresi dat při zachování podstatných informací
- Mohou sloužit pro generování nových dat
- Často se používají pro redukci dimenzionality a detekci anomálií

## Základní architektura

Autoenkodér se skládá ze dvou hlavních komponent:

1. **Encoder (kodér)** - převádí vstup na kompaktní reprezentaci (latentní vektor)
2. **Decoder (dekodér)** - rekonstruuje původní vstupní data z latentní reprezentace

Architektura lze znázornit následovně:

```
Vstup → [Encoder] → Latentní reprezentace → [Decoder] → Rekonstruovaný výstup
```

Během trénování se autoenkodér snaží minimalizovat rekonstrukční chybu mezi originálním vstupem a jeho rekonstrukcí. Typicky se používá střední kvadratická chyba (MSE) nebo binární cross-entropy jako ztrátová funkce.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Načtení a příprava MNIST datasetu
(x_train, _), (x_test, _) = mnist.load_data()

# Normalizace a reshape dat
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Definice parametrů
input_dim = 784  # 28x28 pixelů
encoding_dim = 32  # Dimenze latentní reprezentace

# Definice architektury autoenkodéru
input_layer = Input(shape=(input_dim,))
# Encoder
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(encoding_dim, activation='relu')(encoded)
# Decoder
decoded = Dense(64, activation='relu')(latent)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

# Vytvoření modelu
autoencoder = Model(input_layer, output_layer)

# Kompilace modelu
autoencoder.compile(optimizer='adam', loss='mse')

# Shrnutí architektury
autoencoder.summary()

# Trénování modelu
history = autoencoder.fit(
    x_train, x_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Vykreslení průběhu trénování
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Trénovací ztráta')
plt.plot(history.history['val_loss'], label='Validační ztráta')
plt.title('Průběh trénování autoenkodéru')
plt.ylabel('Ztráta (MSE)')
plt.xlabel('Epocha')
plt.legend()
plt.grid(True)
plt.show()

# Vytvoření encoderu a decoderu pro pozdější použití
encoder = Model(input_layer, latent)
encoded_input = Input(shape=(encoding_dim,))
decoder_layers = autoencoder.layers[-3:] 
decoder = Model(encoded_input, decoder_layers[-1](decoder_layers[-2](decoder_layers[-3](encoded_input))))

# Vizualizace rekonstrukcí
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Náhodný výběr obrázku z testovacího datasetu
    img = x_test[i].reshape(1, -1)
    # Kódování a následná rekonstrukce
    encoded_img = encoder.predict(img)
    decoded_img = decoder.predict(encoded_img)
    
    # Zobrazení
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Originál")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_img.reshape(28, 28), cmap='gray')
    plt.title("Rekonstrukce")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
```

## Typy autoenkodérů

### Vanilla autoenkodéry

Základní typ autoenkodéru popsaný výše, který používá plně propojené vrstvy pro kodér i dekodér. Vhodný především pro jednodušší data.

### Sparse autoenkodéry

Sparse (řídké) autoenkodéry přidávají regularizaci, která nutí síť využívat jen omezený počet neuronů v latentní vrstvě pro reprezentaci dat. Tento přístup může vést k učení významnějších rysů v datech.

```python
from tensorflow.keras.regularizers import l1

# Definice parametrů
input_dim = 784
encoding_dim = 64

# Sparse autoenkodér s L1 regularizací
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu', 
                activity_regularizer=l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Vytvoření modelu
sparse_autoencoder = Model(input_layer, decoded)

# Kompilace modelu
sparse_autoencoder.compile(optimizer='adam', loss='mse')

# Trénování modelu
sparse_autoencoder.fit(
    x_train, x_train,
    epochs=15,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)
```

### Denoising autoenkodéry

Denoising (odšumovací) autoenkodéry se učí rekonstruovat čistá data z jejich zašuměných verzí. Cílem je zvýšit robustnost modelu a naučit ho rozpoznávat skutečný signál od šumu.

```python
import numpy as np

# Funkce pro přidání šumu do obrázků
def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy_images, 0.0, 1.0)

# Vytvoření zašuměných verzí trénovacích a testovacích dat
x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

# Definice architektury denoising autoenkodéru
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(latent)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

# Vytvoření modelu
denoising_autoencoder = Model(input_layer, output_layer)

# Kompilace modelu
denoising_autoencoder.compile(optimizer='adam', loss='mse')

# Trénování modelu - vstup jsou zašuměná data, výstup čistá data
history = denoising_autoencoder.fit(
    x_train_noisy, x_train,
    epochs=15,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# Vizualizace výsledků
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # Výběr náhodného obrázku
    img = x_test_noisy[i].reshape(1, -1)
    # Rekonstrukce
    decoded_img = denoising_autoencoder.predict(img)
    
    # Zobrazení
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Originál")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title("Zašuměný")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_img.reshape(28, 28), cmap='gray')
    plt.title("Rekonstrukce")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
```

### Variační autoenkodéry (VAE)

Variační autoenkodéry (VAE) jsou pravděpodobnostní modely, které místo jediného bodu v latentním prostoru generují distribuce (typicky normální). To umožňuje generovat nové vzorky a lépe modelovat pravděpodobnostní strukturu dat.

```python
import tensorflow as tf
from tensorflow.keras.layers import Lambda

# Definice parametrů
latent_dim = 2  # Nízká dimenze pro vizualizaci
input_dim = 784

# Definice enkodéru
inputs = Input(shape=(input_dim,))
x = Dense(256, activation='relu')(inputs)
x = Dense(128, activation='relu')(x)

# Parametry latentního prostoru - střední hodnoty a logaritmy rozptylů
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Sampling funkce pro reparametrizační trik
def sampling(args):
    z_mean, z_log_var = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Vrstva pro sampling
z = Lambda(sampling)([z_mean, z_log_var])

# Definice dekodéru
decoder_input = Input(shape=(latent_dim,))
x = Dense(128, activation='relu')(decoder_input)
x = Dense(256, activation='relu')(x)
outputs = Dense(input_dim, activation='sigmoid')(x)

# Definice enkodéru a dekodéru jako samostatných modelů
encoder = Model(inputs, [z_mean, z_log_var, z])
decoder = Model(decoder_input, outputs)

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs)

# KL divergence loss
kl_loss = -0.5 * tf.reduce_mean(
    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

# Rekonstrukční loss
reconstruction_loss = tf.reduce_mean(
    tf.keras.losses.binary_crossentropy(inputs, outputs)) * input_dim

# Celková loss funkce
vae_loss = reconstruction_loss + kl_loss

# Přidání loss jako metriku pro sledování
vae.add_loss(vae_loss)
vae.add_metric(kl_loss, name='kl_loss')
vae.add_metric(reconstruction_loss, name='reconstruction_loss')

# Kompilace modelu
vae.compile(optimizer='adam')

# Trénování VAE
history = vae.fit(
    x_train, None,  # nepotřebujeme výstupní data, protože loss je počítána interně
    epochs=15,
    batch_size=128,
    validation_data=(x_test, None)
)

# Vizualizace latentního prostoru
def plot_latent_space(encoder, decoder):
    # Zobrazení latentního prostoru
    n = 15  # počet obrázků podél každé osy
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Pravidelná mřížka bodů v latentním prostoru
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    
    # Generování obrázků z bodů v latentním prostoru
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.title('2D latentní prostor (interpolace mezi číslicemi)')
    plt.axis('off')
    plt.show()

# Vizualizace latentního prostoru
plot_latent_space(encoder, decoder)
```

### Konvoluční autoenkodéry

Konvoluční autoenkodéry používají konvoluční vrstvy místo plně propojených vrstev, což je činí vhodnými pro zpracování obrazů. Zachovávají prostorové vztahy v datech a výrazně snižují počet parametrů.

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

# Příprava dat pro konvoluční síť
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Definice architektury konvolučního autoenkodéru
input_img = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)  # 7x7x16

# Decoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Sestavení modelu
conv_autoencoder = Model(input_img, decoded)

# Kompilace modelu
conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Trénování modelu
history = conv_autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Vizualizace rekonstrukcí
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Výběr obrázku a jeho rekonstrukce
    img = x_test[i:i+1]
    decoded_img = conv_autoencoder.predict(img)
    
    # Zobrazení
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Originál")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_img[0].reshape(28, 28), cmap='gray')
    plt.title("Rekonstrukce")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
```

## Implementace v deep learning frameworcích

### TensorFlow/Keras

TensorFlow s Keras API nabízí vysokoúrovňový a intuitivní způsob vytváření autoenkodérů. Příklad výše ukazuje implementaci v TensorFlow/Keras.

### PyTorch

PyTorch poskytuje flexibilnější a nízkoúrovňovější přístup k vytváření autoenkodérů.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Nastavení zařízení
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Načtení a příprava dat
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Definice autoenkodéru v PyTorch
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Inicializace modelu, loss funkce a optimalizátoru
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Funkce pro trénování modelu
def train(model, dataloader, criterion, optimizer):
    model.train()
    train_loss = 0
    for data, _ in dataloader:
        data = data.to(device)
        # Flatten
        data_flat = data.view(data.size(0), -1)
        # Forward pass
        output = model(data)
        loss = criterion(output, data_flat)
        # Backward pass a optimalizace
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

# Funkce pro testování modelu
def test(model, dataloader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            # Flatten
            data_flat = data.view(data.size(0), -1)
            # Forward pass
            output = model(data)
            loss = criterion(output, data_flat)
            test_loss += loss.item()
    return test_loss / len(dataloader)

# Trénování modelu
num_epochs = 10
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss = test(model, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

# Vizualizace průběhu trénování
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Trénovací ztráta')
plt.plot(test_losses, label='Testovací ztráta')
plt.title('Průběh trénování autoenkodéru v PyTorch')
plt.xlabel('Epocha')
plt.ylabel('Ztráta (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Vizualizace rekonstrukcí
def visualize_reconstructions(model, dataloader):
    model.eval()
    with torch.no_grad():
        # Získání jedné dávky dat
        images, _ = next(iter(dataloader))
        images = images[:10].to(device)  # Vybereme prvních 10 obrázků
        
        # Rekonstrukce
        reconstructions = model(images).view(-1, 28, 28).cpu().numpy()
        images = images.view(-1, 28, 28).cpu().numpy()
        
        # Vizualizace
        plt.figure(figsize=(20, 4))
        for i in range(10):
            # Originál
            ax = plt.subplot(2, 10, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.title("Originál")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            # Rekonstrukce
            ax = plt.subplot(2, 10, i + 11)
            plt.imshow(reconstructions[i], cmap='gray')
            plt.title("Rekonstrukce")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
        plt.tight_layout()
        plt.show()

# Vizualizace výsledků
visualize_reconstructions(model, test_loader)
```

## Aplikace autoenkodérů

### Redukce dimenzionality

Autoenkodéry mohou být použity jako alternativa k tradičním metodám redukce dimenzionality, jako je PCA. Výhodou je, že dokáží zachytit i nelineární vztahy v datech.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Načtení MNIST datasetu a příprava dat
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Definice parametrů
input_dim = 784  # 28x28 pixelů
encoding_dim = 32  # Dimenze latentní reprezentace

# Definice architektury autoenkodéru
input_layer = Input(shape=(input_dim,))
# Encoder
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(encoding_dim, activation='relu')(encoded)
# Vytvoření encoder modelu
encoder = Model(input_layer, latent)

# Kódování dat
encoded_data = encoder.predict(x_test[:1000])  # Použijeme prvních 1000 testovacích obrázků

# Použití t-SNE pro vizualizaci 32D latentního prostoru do 2D
tsne = TSNE(n_components=2, random_state=42)
encoded_data_2d = tsne.fit_transform(encoded_data)

# Vizualizace t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(encoded_data_2d[:, 0], encoded_data_2d[:, 1], c=y_test[:1000], cmap='tab10')
plt.colorbar(scatter, label='Číslice')
plt.title('Vizualizace latentního prostoru autoenkodéru pomocí t-SNE')
plt.xlabel('t-SNE dimenze 1')
plt.ylabel('t-SNE dimenze 2')
plt.show()
```

### Detekce anomálií

Autoenkodéry lze využít pro detekci anomálií tím, že je natrénujeme na normálních datech. Při rekonstrukci anomálních dat bude rekonstrukční chyba výrazně vyšší.

```python
# Načtení MNIST datasetu a příprava dat
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Vytvoření datasetu pouze s jednou číslicí jako "normální" data
normal_digit = 1
x_train_normal = x_train[y_train == normal_digit]
x_test_normal = x_test[y_test == normal_digit]
x_test_anomalies = x_test[y_test != normal_digit]

# Definice architektury autoenkodéru pro detekci anomálií
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

anomaly_autoencoder = Model(input_layer, output_layer)
anomaly_autoencoder.compile(optimizer='adam', loss='mse')

# Trénování pouze na normálních datech
anomaly_autoencoder.fit(
    x_train_normal, x_train_normal,
    epochs=15,
    batch_size=128,
    validation_data=(x_test_normal, x_test_normal)
)

# Výpočet rekonstrukční chyby
def compute_reconstruction_error(model, data):
    reconstructions = model.predict(data)
    mse = np.mean(np.square(data - reconstructions), axis=1)
    return mse

# Výpočet rekonstrukční chyby pro normální a anomální data
normal_reconstruction_error = compute_reconstruction_error(anomaly_autoencoder, x_test_normal)
anomaly_reconstruction_error = compute_reconstruction_error(anomaly_autoencoder, x_test_anomalies[:len(normal_reconstruction_error)])  # Stejný počet vzorků

# Vizualizace distribuce rekonstrukční chyby
plt.figure(figsize=(12, 6))
plt.hist(normal_reconstruction_error, bins=50, alpha=0.5, label=f'Normální data (číslice {normal_digit})')
plt.hist(anomaly_reconstruction_error, bins=50, alpha=0.5, label=f'Anomálie (ostatní číslice)')
plt.legend()
plt.title('Distribuce rekonstrukční chyby')
plt.xlabel('Rekonstrukční chyba (MSE)')
plt.ylabel('Počet vzorků')
plt.yscale('log')  # Logaritmická škála pro lepší vizualizaci
plt.grid(True)
plt.show()

# Stanovení prahu pro detekci anomálií
threshold = np.percentile(normal_reconstruction_error, 95)  # 95. percentil normálních dat

# Aplikace prahu na testovací data
predictions = compute_reconstruction_error(anomaly_autoencoder, x_test[:1000]) > threshold
actual = y_test[:1000] != normal_digit

# Výpočet metrik
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(actual, predictions)
precision = precision_score(actual, predictions)
recall = recall_score(actual, predictions)
f1 = f1_score(actual, predictions)
cm = confusion_matrix(actual, predictions)

print(f"Přesnost: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# Vizualizace několika příkladů
plt.figure(figsize=(20, 8))
for i in range(10):
    # Původní obrázek
    ax = plt.subplot(3, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Číslice: {y_test[i]}")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Rekonstrukce
    reconstruction = anomaly_autoencoder.predict(x_test[i:i+1])
    mse = np.mean(np.square(x_test[i] - reconstruction[0]))
    
    ax = plt.subplot(3, 10, i + 11)
    plt.imshow(reconstruction[0].reshape(28, 28), cmap='gray')
    plt.title(f"Rekon.")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Rozdílový obrázek
    ax = plt.subplot(3, 10, i + 21)
    difference = np.abs(x_test[i] - reconstruction[0])
    plt.imshow(difference.reshape(28, 28), cmap='hot')
    is_anomaly = "Ano" if mse > threshold else "Ne"
    plt.title(f"MSE: {mse:.3f}\nAnomálie: {is_anomaly}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
```

### Odstranění šumu z obrázků

Denoising autoenkodéry mohou efektivně odstraňovat šum z obrázků tím, že se učí rekonstruovat čistá data ze zašuměných vstupů.

```python
from tensorflow.keras.datasets import fashion_mnist

# Načtení Fashion MNIST datasetu pro změnu
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# Funkce pro přidání šumu (Gaussian + Salt & Pepper)
def add_mixed_noise(images, gaussian_factor=0.2, salt_pepper_factor=0.1):
    # Gaussian šum
    noisy_images = images + gaussian_factor * np.random.normal(
        loc=0.0, scale=1.0, size=images.shape)
    
    # Salt & Pepper šum
    mask = np.random.random(size=images.shape) < salt_pepper_factor
    noisy_images[mask] = 1.0  # Salt šum (bílé pixely)
    
    mask = np.random.random(size=images.shape) < salt_pepper_factor
    noisy_images[mask] = 0.0  # Pepper šum (černé pixely)
    
    return np.clip(noisy_images, 0.0, 1.0)

# Vytvoření zašuměných verzí trénovacích a testovacích dat
x_train_noisy = add_mixed_noise(x_train)
x_test_noisy = add_mixed_noise(x_test)

# Definice architektury konvolučního denoising autoenkodéru
input_img = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Sestavení modelu
denoising_conv_autoencoder = Model(input_img, decoded)

# Kompilace modelu
denoising_conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
denoising_conv_autoencoder.summary()

# Trénování modelu
history = denoising_conv_autoencoder.fit(
    x_train_noisy, x_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test_noisy, x_test)
)

# Vizualizace průběhu trénování
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Trénovací ztráta')
plt.plot(history.history['val_loss'], label='Validační ztráta')
plt.title('Průběh trénování denoising autoenkodéru')
plt.ylabel('Ztráta (Binary Cross-Entropy)')
plt.xlabel('Epocha')
plt.legend()
plt.grid(True)
plt.show()

# Vizualizace výsledků odstranění šumu
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # Výběr náhodného obrázku
    img_idx = np.random.randint(0, len(x_test))
    clean_img = x_test[img_idx:img_idx+1]
    noisy_img = x_test_noisy[img_idx:img_idx+1]
    
    # Rekonstrukce
    denoised_img = denoising_conv_autoencoder.predict(noisy_img)
    
    # Zobrazení
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(clean_img[0].reshape(28, 28), cmap='gray')
    plt.title("Originál")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(noisy_img[0].reshape(28, 28), cmap='gray')
    plt.title("Zašuměný")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(denoised_img[0].reshape(28, 28), cmap='gray')
    plt.title("Odšuměný")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
```

### Generativní modely

Variační autoenkodéry jsou typ generativních modelů, které mohou generovat nová data vzorkováním z latentního prostoru.

```python
# Definice VAE
latent_dim = 2  # Nízká dimenze pro lepší vizualizaci

# Encoder
inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)

z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Sampling funkce
def sampling(args):
    z_mean, z_log_var = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Vrstva pro sampling
z = Lambda(sampling)([z_mean, z_log_var])

# Sestavení encoderu
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# Dekodér
latent_inputs = Input(shape=(latent_dim,))
x = Dense(7 * 7 * 64, activation='relu')(latent_inputs)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
x = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
outputs = Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)

# Sestavení dekoderu
decoder = Model(latent_inputs, outputs, name='decoder')

# Sestavení VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# Custom loss funkce
reconstruction_loss = tf.reduce_mean(
    tf.keras.losses.binary_crossentropy(inputs, outputs)) * 28 * 28
kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Trénování VAE
vae.fit(
    x_train, None, 
    epochs=10, 
    batch_size=128, 
    validation_data=(x_test, None)
)

# Generování nových obrázků z latentního prostoru
n = 15  # počet obrázků podél každé osy
figure = np.zeros((28 * n, 28 * n))

# Pravidelná mřížka bodů v latentním prostoru
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.title('Generování obrázků pomocí VAE')
plt.axis('off')
plt.tight_layout()
plt.show()

# Vizualizace latentního prostoru
n_samples = 5000
x_test_sample = x_test[:n_samples]
y_test_sample = y_test[:n_samples]

z_mean, z_log_var, z = encoder.predict(x_test_sample)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(z[:, 0], z[:, 1], c=y_test_sample, cmap='tab10', alpha=0.8)
plt.colorbar(scatter, label='Třída')
plt.title('Latentní prostor VAE')
plt.xlabel('z[0]')
plt.ylabel('z[1]')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Doporučovací systémy

Autoenkodéry se používají i v doporučovacích systémech, kde dokáží předpovědět uživatelské preference na základě neúplných dat.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Vygenerování syntetických dat pro ukázku (matice uživatelé x filmy s hodnoceními)
np.random.seed(42)
n_users = 1000
n_movies = 50
sparsity = 0.8  # 80% hodnot bude chybět

# Vytvoření skutečné matice preferencí (nízké hodnosti)
true_rank = 10
U = np.random.normal(0, 1, (n_users, true_rank))
V = np.random.normal(0, 1, (true_rank, n_movies))
true_ratings = np.dot(U, V)

# Přidání šumu
true_ratings += np.random.normal(0, 0.5, true_ratings.shape)

# Omezení hodnocení na škálu 1-5
true_ratings = np.clip(true_ratings, 1, 5)

# Vytvoření masky pro chybějící hodnoty
mask = np.random.binomial(1, 1-sparsity, (n_users, n_movies))
observed_ratings = true_ratings * mask

# Vizualizace dat
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(true_ratings[:20, :20], cmap='viridis', aspect='auto')
plt.colorbar(label='Hodnocení')
plt.title('Skutečná hodnocení (výřez 20x20)')
plt.xlabel('Filmy')
plt.ylabel('Uživatelé')

plt.subplot(1, 3, 2)
plt.imshow(mask[:20, :20], cmap='binary', aspect='auto')
plt.colorbar(label='Maska')
plt.title('Maska dostupných hodnocení')
plt.xlabel('Filmy')
plt.ylabel('Uživatelé')

plt.subplot(1, 3, 3)
plt.imshow(observed_ratings[:20, :20], cmap='viridis', aspect='auto')
plt.colorbar(label='Hodnocení')
plt.title('Pozorovaná hodnocení (s chybějícími)')
plt.xlabel('Filmy')
plt.ylabel('Uživatelé')

plt.tight_layout()
plt.show()

# Rozdělení dat na trénovací a testovací
train_mask = np.random.binomial(1, 0.8, (n_users, n_movies))  # 80% pro trénink
test_mask = np.logical_and(mask, np.logical_not(train_mask))  # Jen dostupná data, která nejsou v tréninku

train_ratings = observed_ratings * train_mask
test_ratings = observed_ratings * test_mask

# Nahrazení chybějících hodnot nulami (pro účely trénování)
train_ratings_zero = np.copy(train_ratings)
train_ratings_zero[train_ratings_zero == 0] = 0

# Vytvoření modelu autoenkodéru pro doporučovací systém
input_dim = n_movies
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
# Encoder
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dropout(0.3)(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dropout(0.3)(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)  # Sigmoid pro normalizaci do [0,1]

# Sestavení modelu
autoencoder = Model(input_layer, output_layer)

# Kompilace modelu s vlastní loss funkcí, která ignoruje chybějící hodnoty
def masked_mse(y_true, y_pred):
    # Vytvoření masky, kde jsou hodnoty > 0
    mask = tf.cast(tf.math.greater(y_true, 0), dtype=tf.float32)
    # Aplikace masky na skutečná a předpovídaná data
    masked_y_true = tf.multiply(y_true, mask)
    masked_y_pred = tf.multiply(y_pred, mask)
    # Výpočet MSE pouze pro nechybějící hodnoty
    loss = tf.reduce_sum(tf.square(masked_y_true - masked_y_pred)) / tf.reduce_sum(mask)
    return loss

# Normalizace hodnocení do rozsahu [0,1] pro použití se sigmoid aktivací
max_rating = 5.0
train_normalized = train_ratings_zero / max_rating
test_normalized = test_ratings / max_rating

# Kompilace modelu
autoencoder.compile(optimizer='adam', loss=masked_mse)

# Trénování modelu
history = autoencoder.fit(
    train_normalized, train_normalized,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Vizualizace průběhu trénování
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Trénovací ztráta')
plt.plot(history.history['val_loss'], label='Validační ztráta')
plt.title('Průběh trénování autoenkodéru pro doporučování')
plt.xlabel('Epocha')
plt.ylabel('Ztráta (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Predikce všech hodnocení
predicted_ratings_normalized = autoencoder.predict(train_normalized)
predicted_ratings = predicted_ratings_normalized * max_rating

# Vytvoření masky, kde jsou testovací hodnoty
test_indices = np.where(test_mask)
test_predictions = predicted_ratings[test_indices]
test_true = true_ratings[test_indices]

# Výpočet RMSE na testovacích datech
rmse = np.sqrt(mean_squared_error(test_true, test_predictions))
print(f"RMSE na testovacích datech: {rmse:.4f}")

# Vizualizace skutečných vs. předpovězených hodnocení
plt.figure(figsize=(10, 6))
plt.scatter(test_true, test_predictions, alpha=0.1)
plt.plot([1, 5], [1, 5], 'r--')  # Ideální předpověď
plt.xlabel('Skutečná hodnocení')
plt.ylabel('Předpovězená hodnocení')
plt.title('Skutečná vs. předpovězená hodnocení')
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.grid(True)
plt.show()

# Vizualizace doporučení pro konkrétního uživatele
user_id = 10
user_ratings = observed_ratings[user_id]
user_mask = mask[user_id]

# Filmy, které uživatel již hodnotil
rated_movies = np.where(user_mask)[0]
unrated_movies = np.where(np.logical_not(user_mask))[0]

# Získání předpovědí pro všechny filmy
user_predictions = predicted_ratings[user_id]

# Top-5 doporučení mezi nehodnocenými filmy
top_recommendations = unrated_movies[np.argsort(-user_predictions[unrated_movies])[:5]]

print(f"Uživatel {user_id} - hodnocené filmy:")
for movie in rated_movies:
    print(f"  Film {movie}: {observed_ratings[user_id, movie]:.1f}")

print("\nTop-5 doporučení:")
for movie in top_recommendations:
    print(f"  Film {movie}: {user_predictions[movie]:.1f} (předpověď)")
```

## Pokročilé techniky a variace

### Conditional VAEs

Conditional Variational Autoencoders (CVAE) jsou rozšířením standardních VAE, které umožňují podmíněné generování dat. Do modelu je přidána další podmínka (např. třída, label), která řídí proces generování.

#### Princip funkce CVAE:
- Přidává podmínkovou informaci (např. label třídy) do encoderu i decoderu
- Dovoluje kontrolovat generativní proces pomocí této podmínky
- Zlepšuje kvalitu generování pro specifické třídy nebo vlastnosti

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Načtení MNIST datasetu
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Předzpracování dat
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# One-hot kódování tříd
y_train_cat = tf.keras.utils.to_categorical(y_train)
y_test_cat = tf.keras.utils.to_categorical(y_test)

# Parametry modelu
original_dim = 784  # 28*28 pixelů
latent_dim = 2      # Dimenze latentního prostoru
intermediate_dim = 256
num_classes = 10    # MNIST má 10 tříd

# Vytvoření modelu CVAE
# Encoder
inputs = Input(shape=(original_dim,), name='encoder_input')
label_inputs = Input(shape=(num_classes,), name='class_input')
x = Concatenate()([inputs, label_inputs])
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(h)
z_log_var = Dense(latent_dim, name='z_log_var')(h)

# Sampling funkce
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
label_inputs_decoder = Input(shape=(num_classes,), name='class_input_decoder')
z_cat = Concatenate()([latent_inputs, label_inputs_decoder])
decoder_h = Dense(intermediate_dim, activation='relu')(z_cat)
outputs = Dense(original_dim, activation='sigmoid')(decoder_h)

# Definice encoderu
encoder = Model([inputs, label_inputs], [z_mean, z_log_var, z], name='encoder')

# Definice decoderu
decoder = Model([latent_inputs, label_inputs_decoder], outputs, name='decoder')

# Definice CVAE modelu
outputs = decoder([encoder([inputs, label_inputs])[2], label_inputs])
cvae = Model([inputs, label_inputs], outputs, name='cvae')

# Loss funkce pro VAE
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=-1)
    return xent_loss + kl_loss

cvae.compile(optimizer='adam', loss=vae_loss)

# Trénování modelu
cvae.fit([x_train, y_train_cat], x_train,
         epochs=20,
         batch_size=128,
         validation_data=([x_test, y_test_cat], x_test))

# Funkce pro generování číslic dané třídy
def generate_digits(model, latent_dim, digit_class, num_samples=10):
    # Generování náhodných bodů v latentním prostoru
    z_sample = np.random.normal(0, 1, size=(num_samples, latent_dim))
    
    # Vytvoření one-hot encoded tříd
    digit_to_generate = np.zeros((num_samples, num_classes))
    digit_to_generate[:, digit_class] = 1
    
    # Generování čísel pomocí decoderu
    generated = decoder.predict([z_sample, digit_to_generate])
    
    # Vizualizace
    plt.figure(figsize=(15, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(generated[i].reshape(28, 28), cmap='Greys_r')
        plt.axis('off')
    plt.suptitle(f'Generované číslice třídy {digit_class}')
    plt.tight_layout()
    plt.show()

# Generování číslic pro každou třídu
for i in range(10):
    generate_digits(cvae, latent_dim, i)
```

### β-VAE

β-VAE je rozšíření variačních autoencoder sítí, které klade větší důraz na disentangled reprezentace. Přidává hyperparametr β, který váží KL divergenci v loss funkci.

#### Klíčové vlastnosti β-VAE:
- Umožňuje vytvářet lépe rozpletené (disentangled) reprezentace
- Vyšší hodnoty β podporují nezávislejší latentní faktory
- Lepší interpretovatelnost latentního prostoru

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Načtení MNIST datasetu
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Předzpracování dat
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Parametry modelu
original_dim = 784  # 28*28 pixelů
intermediate_dim = 256
latent_dim = 10     # Větší latentní prostor pro lepší disentanglement
beta = 10           # Váha KL divergence (β hodnota)

# Encoder
inputs = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Sampling funkce
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_mean(h_decoded)

# Definice β-VAE modelu
beta_vae = Model(inputs, outputs)

# Encoder model
encoder = Model(inputs, [z_mean, z_log_var, z])

# Decoder model
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_outputs = decoder_mean(_h_decoded)
decoder = Model(decoder_input, _outputs)

# β-VAE loss funkce s β parametrem
def beta_vae_loss(x, x_decoded_mean):
    reconstruction_loss = original_dim * tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=-1)
    return reconstruction_loss + beta * kl_loss

beta_vae.compile(optimizer='adam', loss=beta_vae_loss)

# Trénování modelu
beta_vae.fit(x_train, x_train,
             epochs=30,
             batch_size=128,
             validation_data=(x_test, x_test))

# Vizualizace latentního prostoru
n = 10  # Počet digitů, které chceme analyzovat
digit_size = 28

# Vizualizace latentních dimenzí
def plot_latent_space(encoder, decoder, n_grid=10):
    # Zobrazení několika dimenzí latentního prostoru
    for dim1 in range(3):  # Zobrazíme první 3 dimenze
        for dim2 in range(dim1 + 1, 4):  # a jejich vzájemné kombinace
            figure = np.zeros((digit_size * n_grid, digit_size * n_grid))
            
            # Vytvoření hodnot pro mřížku
            grid_x = np.linspace(-3, 3, n_grid)
            grid_y = np.linspace(-3, 3, n_grid)[::-1]
            
            # Procházení mřížkou
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = np.zeros((1, latent_dim))
                    z_sample[0, dim1] = xi
                    z_sample[0, dim2] = yi
                    x_decoded = decoder.predict(z_sample)
                    digit = x_decoded[0].reshape(digit_size, digit_size)
                    figure[i * digit_size: (i + 1) * digit_size,
                           j * digit_size: (j + 1) * digit_size] = digit
            
            plt.figure(figsize=(10, 10))
            plt.imshow(figure, cmap='Greys_r')
            plt.title(f'Latentní dimenze {dim1} vs {dim2}')
            plt.show()

# Vizualizace latentního prostoru
plot_latent_space(encoder, decoder)
```

### VQ-VAE

Vector Quantized Variational Autoencoders (VQ-VAE) kombinují VAE s vektorovou kvantizací, což umožňuje diskrétní latentní reprezentace. Používají codebook vektorů pro mapování z kontinuálního do diskrétního prostoru.

#### Klíčové vlastnosti VQ-VAE:
- Diskrétní latentní prostor místo spojitého
- Používá vektorovou kvantizaci pro mapování na diskrétní codewordy
- Lépe modeluje komplexní multimodální distribuce
- Slouží jako základ pro pokročilé generativní modely jako VQ-VAE-2 a VQ-GAN

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Načtení MNIST datasetu
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Předzpracování dat
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Parametry modelu
embedding_dim = 64  # Dimenze embedding vektorů
num_embeddings = 512  # Počet vektorů v codeknize
commitment_cost = 0.25  # Váha commitment loss

# Vytvoření VQ-VAE

# Encoder
input_img = Input(shape=(28, 28, 1))
x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
x = Conv2D(embedding_dim, 1, padding='same')(x)
encoder_output_shape = tf.keras.backend.int_shape(x)
encoder_output = x

# Vector Quantizer
class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(self.embedding_dim, self.num_embeddings),
                                 initializer='uniform',
                                 trainable=True,
                                 name='embeddings')
        
    def call(self, inputs):
        # Reshape vstupu pro výpočet vzdáleností
        input_shape = tf.shape(inputs)
        flattened = tf.reshape(inputs, [-1, self.embedding_dim])
        
        # Výpočet vzdáleností ke všem embeddings
        distances = (tf.reduce_sum(flattened**2, axis=1, keepdims=True) 
                    - 2 * tf.matmul(flattened, self.w)
                    + tf.reduce_sum(self.w**2, axis=0, keepdims=True))
        
        # Encoder výstup -> indexy nejbližších embeddingů
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, tf.transpose(self.w))
        quantized = tf.reshape(quantized, input_shape)
        
        # Commitment loss
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Průchod gradientu (Straight-through estimator)
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        
        # Přidání loss jako custom loss
        self.add_loss(loss)
        
        return quantized

quantized_encoder_output = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)(encoder_output)

# Decoder
x = Conv2D(64, 3, padding='same', activation='relu')(quantized_encoder_output)
x = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
x = Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')(x)
decoded = x

# Vytvoření modelu
vqvae = Model(input_img, decoded)
vqvae.compile(optimizer='adam', loss='binary_crossentropy')

# Trénování modelu
vqvae.fit(x_train, x_train,
          epochs=10,
          batch_size=128,
          validation_data=(x_test, x_test))

# Vytvoření encoderu a decoderu pro generování
encoder = Model(input_img, encoder_output)

decoder_input = Input(shape=encoder_output_shape[1:])
quantized_input = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)(decoder_input)
x = Conv2D(64, 3, padding='same', activation='relu')(quantized_input)
x = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
x = Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')(x)
decoder = Model(decoder_input, x)

# Testování rekonstrukce
n = 10
original_images = x_test[:n]
encoded_imgs = encoder.predict(original_images)
decoded_imgs = decoder.predict(encoded_imgs)

# Vizualizace rekonstrukcí
plt.figure(figsize=(20, 4))
for i in range(n):
    # Původní obrázek
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
    plt.title("Originál")
    plt.axis('off')
    
    # Rekonstrukce
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Rekonstrukce")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

## Nejlepší postupy a omezení

### Nejlepší postupy při práci s autoencoder modely

1. **Architektura sítě:**
   - Začněte s jednoduchou architekturou a postupně přidávejte komplexitu
   - Použijte symetrické architektury pro encoder a decoder
   - Pro obrázky jsou vhodnější konvoluční vrstvy, pro sekvence rekurentní vrstvy

2. **Dimenze latentního prostoru:**
   - Menší latentní prostor vede k lepší kompresi, ale horší rekonstrukci
   - Větší latentní prostor umožňuje lepší rekonstrukci, ale horší regularizaci
   - Velikost latentního prostoru volte podle složitosti dat a požadované komprese

3. **Regularizace:**
   - Pro VAE je klíčová správná váha KL divergence (β parametr)
   - Příliš silná regularizace vede k rozmazaným rekonstrukcím
   - Příliš slabá regularizace vede ke špatné generativní schopnosti

4. **Trénink:**
   - Používejte dávkovou normalizaci pro stabilnější trénink
   - Pro VAE zvažte postupné zvyšování váhy KL divergence (annealing)
   - U VQ-VAE věnujte pozornost parametru commitment_cost

5. **Evaluace:**
   - Sledujte rekonstrukční chybu na validačních datech
   - Pro VAE sledujte také KL divergenci
   - Vizuálně kontrolujte kvalitu rekonstrukcí a generovaných vzorků

```python
# Příklad implementace KL annealing pro VAE
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback

class KLAnnealingCallback(Callback):
    def __init__(self, beta_start=0, beta_end=1, epochs=10):
        super(KLAnnealingCallback, self).__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epochs = epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        # Lineární zvyšování beta parametru
        beta = self.beta_start + (epoch / self.epochs) * (self.beta_end - self.beta_start)
        beta = min(beta, self.beta_end)  # Ujištění, že nepřekročíme max hodnotu
        
        # Aktualizace beta parametru v loss funkci
        # Poznámka: Toto vyžaduje, aby loss funkce byla dynamicky upravitelná
        tf.keras.backend.set_value(self.model.beta, beta)
        print(f"Epoch {epoch}: beta = {beta}")

# Vytvoření VAE modelu s annealing
def create_vae_with_annealing(input_dim, latent_dim):
    # Encoder
    inputs = Input(shape=(input_dim,))
    h = Dense(256, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    # Beta parametr jako proměnná modelu
    beta = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='beta')
    
    # Sampling funkce
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling)([z_mean, z_log_var])
    
    # Decoder
    decoder_h = Dense(256, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    outputs = decoder_mean(h_decoded)
    
    # VAE model
    vae = Model(inputs, outputs)
    
    # Custom loss funkce s dynamickým beta parametrem
    def vae_loss(x, x_decoded_mean):
        xent_loss = input_dim * tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1)
        return xent_loss + beta * kl_loss
    
    vae.compile(optimizer='adam', loss=vae_loss)
    
    # Uložení beta jako atribut modelu pro callback
    vae.beta = beta
    
    return vae

# Použití:
# model = create_vae_with_annealing(784, 10)
# kl_annealing = KLAnnealingCallback(beta_start=0.0, beta_end=5.0, epochs=20)
# model.fit(x_train, x_train, epochs=30, batch_size=128, callbacks=[kl_annealing])
```

### Omezení autoencoderů

1. **Omezení rekonstrukční kvality:**
   - Tradiční autoencodery často produkují rozmazané rekonstrukce
   - Komprese do malého latentního prostoru způsobuje ztrátu detailů

2. **Výzvy v latentním prostoru:**
   - U standardních autoencoder nelze snadno vzorkovat z latentního prostoru
   - U VAE může být latentní prostor příliš regularizovaný a vést ke ztrátě informací
   - VQ-VAE vyžaduje komplexní architekturu a správné nastavení codebooku

3. **Tréninkové výzvy:**
   - Hledání správné rovnováhy mezi rekonstrukční přesností a regularizací
   - U VAE je obtížné nalézt ideální hodnotu β parametru
   - VQ-VAE může trpět tzv. "codebook collapse" problémem

4. **Aplikační omezení:**
   - Generované vzorky často nedosahují kvality GAN modelů
   - Složitější varianty autoencoder vyžadují více výpočetních zdrojů
   - Pro velmi vysokorozměrná data může být potřeba hierarchická struktura

```python
# Demonstrace "posterior collapse" problému ve VAE
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

def create_vae(input_dim, latent_dim, beta=1.0):
    inputs = Input(shape=(input_dim,))
    h = Dense(256, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling)([z_mean, z_log_var])
    
    decoder_h = Dense(256, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    outputs = decoder_mean(h_decoded)
    
    vae = Model(inputs, outputs)
    
    # Custom loss s různou váhou KL divergence
    def vae_loss(x, x_decoded_mean):
        xent_loss = input_dim * tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1)
        return xent_loss + beta * kl_loss
    
    vae.compile(optimizer='adam', loss=vae_loss)
    return vae, z_mean, z_log_var

# Vizualizace rozdílných beta hodnot
def visualize_posterior_collapse(x_train, latent_dim=2, epochs=20):
    beta_values = [0.1, 1.0, 10.0]
    reconstruction_loss = {beta: [] for beta in beta_values}
    kl_divergence = {beta: [] for beta in beta_values}
    
    for beta in beta_values:
        print(f"Trénink modelu s beta = {beta}")
        vae, z_mean, z_log_var = create_vae(x_train.shape[1], latent_dim, beta)
        
        # Custom callback pro sledování KL divergence a rekonstrukční chyby
        class MetricsCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                z_mean_val, z_log_var_val = encoder.predict(x_train[:1000])
                kl = -0.5 * np.mean(
                    1 + z_log_var_val - np.square(z_mean_val) - np.exp(z_log_var_val))
                kl_divergence[beta].append(kl)
                
                reconstructions = vae.predict(x_train[:1000])
                mse = np.mean(np.square(x_train[:1000] - reconstructions))
                reconstruction_loss[beta].append(mse)
        
        # Encoder model pro metriky
        encoder = Model(vae.inputs, [z_mean, z_log_var])
        
        # Trénování modelu
        vae.fit(x_train, x_train,
               epochs=epochs,
               batch_size=128,
               verbose=0,
               callbacks=[MetricsCallback()])
    
    # Vizualizace výsledků
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for beta in beta_values:
        plt.plot(reconstruction_loss[beta], label=f'Beta = {beta}')
    plt.title('Rekonstrukční chyba (MSE)')
    plt.xlabel('Epocha')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for beta in beta_values:
        plt.plot(kl_divergence[beta], label=f'Beta = {beta}')
    plt.title('KL Divergence')
    plt.xlabel('Epocha')
    plt.ylabel('KL Divergence')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Použití funkce
# visualize_posterior_collapse(x_train)
```

### Porovnání typů autoencoderů

| Typ | Výhody | Nevýhody | Vhodné aplikace |
|-----|--------|----------|-----------------|
| Základní AE | Jednoduchost, rychlý trénink | Nepravidelný latentní prostor, nelze generovat nové vzorky | Redukce dimenzionality, feature extraction |
| Denoising AE | Robustní features, lepší generalizace | Vyžaduje generování zašuměných dat | Odstranění šumu, robustní feature learning |
| Variační AE (VAE) | Spojitý latentní prostor, generativní model | Rozmazané rekonstrukce, posterior collapse | Generování dat, interpretovatelné latentní faktory |
| β-VAE | Disentangled reprezentace, kontrola latentního prostoru | Obtížné vyvážení rekonstrukce a disentanglement | Učení interpretovatelných faktorů variace |
| VQ-VAE | Ostřejší rekonstrukce, diskrétní latentní prostor | Složitější implementace, codebook collapse | Generování obrazů a řeči ve vysoké kvalitě |
| Conditional VAE | Kontrolované generování podle podmínek | Složitější architektura | Generování dat podle třídy nebo atributů |

Autoencodery jsou mocným nástrojem pro širokou škálu aplikací od redukce dimenzionality po generování nových dat. Správná volba typu autoencoder a jeho architektury závisí na konkrétním úkolu a požadavcích projektu.