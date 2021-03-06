# CarRacingRL
### Mené à contribution égales par G. Grosse, M. Mahaut, B. Nicol

## Problématique

Maintenir à partir d'images une voiture sur une piste dans une simulation, en gérant direction, accélération et freinage.

## Contexte
Projet : Reinforcement Learning  
Spécialité de filière : Intelligence Artificielle  
Ecoles d'ingénieurs associées : ENSC et ENSEIRB
Durée : 1 mois


## Premiers pas

Si il n'y a pas de données dans ./models (le modèle n'est pas entraîné), lancer un entrainement en exécutant train_model.py ou train_model_v2.py.
Si il est entraîné, lancer la simulation en exécutant test_model.py.


## Contenu du projet

Indications générales sur le training
-------------------------------------

Entraine un réseau DQN pour résoudre la tâche CarRacingV0.
Le DQN s'entraîne avec un memory buffer sur 4 frames successives.
Les récompenses encouragent l'accélération maximale.

train_model.py
--------------

Le réseau de neurones DQN estime une Q-valeur pour chaque paire action-état.
Les Q-valeurs sont normalisées avant entraînement.

train_model_v2.py
--------------

Le réseau DQN estime la Q-valeur de chaque action pour un état donnée. Il renvoie donc 12 valeurs pour les 12 actions de l'espace discrétisé des actions.
L'apprentissage se fait sur un batch de transitions issu du memory buffer (état - action - récompense - nouvel état) qui priorise les transitions qui donnent de hautes récompenses.
Gain de performance en vitesse d'exécution.

train_policy_model.py
--------------
Entraine un réseau Acteur / Critique (qui entraine une policy et une estimation de Q-valeurs en parallèle)
Les performances sont difficile à évaluer, sur les 20 épisodes d'entraînement.
Contrairement aux version précédentes, cet entraînement se fait complètement on-policy (pas de memory buffer)

test_model.py
-------------

Lance la simulation en appelant le modèle entraîné à chaque frame pour définir les paramètres de conduite (direction accélération freinage), avec un rendu graphique.


## Résultats

Le dernier modèle entraîné parvient selon les cas à se maintenir sur piste, une nette amélioration par rapport à ses prédécesseurs.


## Travaux intermédiaires

DQN offline avec discrétisation en 12 actions.

Pre-processing des observations : test en N&B et couleurs, sans impact notable.

Optimizer testé en mse, mae et logcosh : logcosh conservé car proposant une meilleure gestion des valeurs extrêmes (outliers).

Activation testée avec tanh, sigmoid, relu, et lin : on a conservé linéaire car évite l'étape de normalisation.

Adaptation du nombre d'épisodes de test par cycle d'entrainement : les performances sont meilleures avec plus de cycles, le nombre d'épisode par cycle semble moins influent.

Test de reward engineering avec multiplication, addition et soustraction de récompenses : conservation de la multiplication selon les circonstances (accélération, maintien sur piste).

Test des Q-values avec et sans mémoire : avec mémoire nous a semblé plus cohérent (alpha=0.8).

Normalisation des Q-values : pas d'augmentation de performance notable, non conservé.

Ajout d'une séparation des données en train+validation au cours de l'entrainement pour réduire le biais d'overfitting sur les métriques.

Ajout d'un bootstrap pendant l'entrainement : au lieu d'une exploration complètement aléatoire, l'agent est encouragé à accélérer 10% du temps.
L'impact n'étant pas visible de manière positive sur l'entrâinement, nous ne l'avons pas concervé.

Limitation de chaque épisode dans le temps (30 à 50s) puis limitation en nombre de récompenses négatives consécutives (meilleurs résultats)

## Pistes additionnelles

Améliorer la priorisation dans le memory buffer.
Augmenter significativement la durée d'entrainement, le nombre de cycles et d'épisodes.
Ajuster les hyper-paramètres du modèle (alpha, epsilon, gamma, eps-decay etc.).
