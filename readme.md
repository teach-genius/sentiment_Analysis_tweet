# Classification de Sentiment de Tweets avec PyTorch

Ce projet démontre comment construire et entraîner un classifieur de sentiment pour des tweets en utilisant PyTorch. Il utilise un dataset de tweets publiquement disponible pour classifier les messages en différentes catégories de sentiment.

## Description du Projet

L'objectif principal de ce projet est de développer un modèle de classification capable d'analyser le texte des tweets et d'attribuer une catégorie de sentiment (positive, négative, neutre, etc., selon le dataset). Le pipeline inclut le chargement des données depuis une source CSV, la vectorisation des textes avec TF-IDF, la création d'un modèle de réseau de neurones simple avec PyTorch, l'entraînement du modèle, et son évaluation.

## Fonctionnalités

- **Chargement des données** : Télécharge un dataset de tweets directement depuis une URL GitHub.
- **Prétraitement de texte** : Utilise `TfidfVectorizer` pour convertir le texte brut en représentations numériques (TF-IDF), filtrant les mots vides en anglais.
- **Encodage des étiquettes** : Convertit les étiquettes de catégorie de texte en entiers numériques.
- **Dataset et DataLoader PyTorch** : Implémente une classe `NewsDataset` personnalisée pour gérer les données et utilise `DataLoader` pour des chargements de lots efficaces pendant l'entraînement.
- **Modèle de réseau de neurones** : Définit un classifieur simple (`NewsClassifier`) avec une couche cachée et une activation ReLU.
- **Entraînement du modèle** : Entraîne le modèle en utilisant l'optimiseur Adam et la fonction de perte Cross-Entropy.
- **Évaluation des performances** : Calcule et affiche un rapport de classification détaillé (précision, rappel, f1-score, support).

## Prérequis

Assurez-vous d'avoir les bibliothèques Python suivantes installées :

- `pandas`
- `numpy`
- `torch`
- `scikit-learn`

Vous pouvez les installer via pip :

```bash
pip install pandas numpy torch scikit-learn
```

## Utilisation

1.  **Exécutez le script :**
    Le script télécharge automatiquement le dataset depuis l'URL spécifiée. Assurez-vous d'avoir une connexion internet active lors de la première exécution.

    ```bash
    python votre_script.py
    ```

    Remplacez `votre_script.py` par le nom de votre fichier Python.

## Exemple de Sortie

Après l'exécution du script, vous verrez l'évolution de la perte d'entraînement par époque, suivie d'un rapport de classification complet sur le jeu de test.

```
Epoch 1, Loss: 89.4223
Epoch 2, Loss: 38.2249
Epoch 3, Loss: 28.7918
Epoch 4, Loss: 23.4632
Epoch 5, Loss: 19.3565
              precision    recall  f1-score   support

           0       0.97      0.98      0.97      5937
           1       0.65      0.60      0.63       456

    accuracy                           0.95      6393
   macro avg       0.81      0.79      0.80      6393
weighted avg       0.95      0.95      0.95      6393
```

**Note sur le dataset :** Le rapport de classification ci-dessus montre des résultats pour les classes 0 et 1. Ces classes correspondent aux labels encodés par `LabelEncoder`. Dans le cas du dataset "Twitter Sentiment Analysis", la classe `0` représente généralement un sentiment négatif/neutre et la classe `1` un sentiment positif. L'imbalance entre les classes (beaucoup plus d'échantillons pour la classe 0 que pour la classe 1) est visible dans la colonne `support`.

## Structure du Code

- `NewsDataset`: Gère le chargement par lots des textes vectorisés et de leurs étiquettes.
- `NewsClassifier`: Le modèle de réseau de neurones.
- `load_data`: Charge le DataFrame depuis l'URL.
- `preprocess_data`: Vectorise les textes et encode les étiquettes.
- `train_model`: Boucle d'entraînement du modèle.
- `evaluate_model`: Évalue le modèle et affiche le rapport de classification.
- Bloc `if __name__ == "__main__":`: Point d'entrée principal du script.

## Contributions

Les contributions sont appréciées ! Si vous avez des suggestions d'amélioration ou rencontrez des problèmes, n'hésitez pas à ouvrir une issue ou à soumettre une pull request.
