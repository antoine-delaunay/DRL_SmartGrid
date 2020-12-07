# DRL SmartGrid

Ce projet a pour but de déterminer une gestion optimale en terme de coût d'une SmartGrid simple en utilisant des algorithmes d'apprentissage par renforcement profond.

Librairies nécessaires à l'exécution de ce projet :

- Pandas
- TensorFlow 2
- Numpy
- Matplotlib

## Exécuter le projet

Le fichier `main.py` fourni un exemple d'utilisation de ce projet.
Pour l'exécuter, il faut se placer à la racine du projet et exécuter la commande suivante :

```shell
python3 main.py
```

Durant l'entraînement, des logs sont générés dans le dossier `/logs` à la racine du projet. TensorBoard permet de les visualiser. Pour cela, toujours à la racine du projet, il suffit d'exécuter la commande :

```shell
tensorboard --logdir=logs
```

A la fin de la phase d'entraînement, les résultats s'affichent. Les performances de l'algorithme DQN sont comparées à celles d'autres stratégies prédéfinies dans le fichier `Analyse.py` (voir la section associée ci-dessous).

Les paramètres `NB_EPISODES`, `NB_STEPS` et `BATCH_SIZE` situés dans ce fichier sont ceux qui ont le plus d'influence sur le temps d'entraînement de l'algorithme DQN.

## Env.py

L'environnement lié au SmartGrid est situé dans ce fichier. On y trouve notamment les paramètres de la batterie, le prétraitement des données de consommation et de production d'électricité, une classe `State` stockant l'état de l'environnement et une fonction `step` qui met à jour cet état en fonction de l'action donnée et des équations régissant l'environnement. Les données de consommation et de production sont situées dans le dossier `Data` à la racine du projet.

Les actions possibles sont également définies dans ce fichier. Celles-ci sont :

- Trade : les surplus et carence d'électricité sont échangés sur le marché. La batterie n'est pas impliquée ici.
- Charge : s'il y a un surplus de production d'électricité, alors on charge la batterie avec.
- Discharge : si la consommation est supérieure à la production, alors on essaie d'utiliser l'énergie présente dans la batterie pour compenser.

## Model.py

L'algorithme DQN est entièrement codé dans ce fichier. Il se décompose en une fonction principale `train` et plusieurs sous-fonctions appelées par cette fonction `train`.

La fonction `train` s'occupe de créer un réseau de neurones séquentiel et de l'entraîner. Les variables décrivant l'évolution du réseau au cours de l'entraînement sont enregistrées dans le dossier `logs` et visualisables avec TensorBoard. Le réseau entraîné est ensuite renvoyé à la fin de l'entraînement. Cette fonction admet de nombreux paramètres dont voici une description :

- `env` : l'environnement sur lequel on veut entraîner l'algorithme.
- `hidden_layers` : une liste contenant le nombre de neurones par couche caché. Les nombres de neurones à l'entrée et à la sortie sont automatiquement déduit de l'environnement fournit.
- `nb_episodes` : nombre d'épisodes d'entraînement. Un épisode correspond à une suite contiguë d'étapes dans l'environnement.
- `nb_steps` : nombre d'étapes par épisode. Une étape correspond à un appel à la fonction step (de manière équivalente une action sur un état) et à un entraînement sur un batch de la replay memory.
- `batch_size` : nombre d'étapes de la replay memory sur lesquelles on entraîne le réseau à chaque nouvelle étape générée.
- `model_name` : nom du model. Ce paramètre caractérise le nom de la sauvegarde du modèle et le nom qui s'affiche dans TensorBoard.
- `save_episode` : le modèle est sauvegardé tous les `save_episode` épisodes. Si ce paramètre est fixé à `None` alors le modèle n'est pas sauvegardé du tout.
- `recup_model` : booléen définissant si on charge ou non un modèle du nom de `model_name` dans les sauvegardes.
- `algo` : ce paramètre a deux valeurs possibles : `"simple"` ou `"double"`. Il définit l'algorithme à utiliser. Le premier correspond à l'algorithme DQN classique, tandis que le second correspond à l'algorithme du double DQN.
- `replay_memory_init_size` : taille initiale de la replay_memory.
- `replay_memory_size` : taille maximale de la replay_memory.
- `update_target_estimator_init` : période de temps initiale durant laquelle le target estimator n'est pas mis à jour.
- `update_target_estimator_max` : période de temps maximal/final durant laquelle le target estimator n'est pas mis à jour.
- `update_target_estimator_epoch` : la variable `update_target_estimator` commmence à `update_target_estimator_init` et termine à `update_target_estimator_max`. `update_target_estimator_epoch` détermine le nombre d'épisodes à faire avant d'augmenter `update_target_estimator`.
- `epsilon_start` : valeur initiale du paramètre `epsilon` dans l'algorithme DQN.
- `epsilon_min` : valeur finale du paramètre `epsilon` dans l'algorithme DQN.
- `epsilon_decay_steps` : nombre d'étapes pour qu'`epsilon` passe de la valeur `epsilon_start` à la valeur `epsilon_min`.

## Analyse.py

Ce fichier a pour but d'analyser les résultats de l'algorithme DQN. Pour cela, on compare les performances obtenues à celles d'algorithmes prédéfinis dans ce fichier.
Ces stratégies sont les suivantes :

- Random : à chaque étape, une action est choisie aléatoirement.
- RandomBattery : à chaque étape, une action parmi charge et discharge est choisie aléatoirement.
- Trade : la batterie n'est jamais utilisée. Le surplus et les carences sont gérées avec le réseau extérieur.
- SmartBattery : s'il y a un surplus d'énergie, on charge la batterie, sinon on la décharge.
- SmartBattery2 : dans le cas où il y a un surplus d'énergie, si la batterie n'est pas pleine, on charge, sinon on vend sur le réseau. En cas de carence en électricité on décharge la batterie. Si cela n'est pas suffisant, on complète avec l'énergie du réseau. Cette stratégie est optimale si le prix de l'électricité du réseau est constant.

Les différentes stratégies sont comparées sur une même suite d'étapes. Les résultats sont ensuite affichés dans 5 figures différentes.

La première affiche pour chaque stratégie, la quantité d'énergie stockée dans la batterie au cours du temps et la quantité d'énergie échangée à chaque étape.

La deuxième affiche à chaque étape l'action effectuée par chacune des stratégies.

La troisième affiche la production et la consommation d'énergie à chaque étape.

La quatrième place dans un même graphique les courbes de coût au cours du temps associées à chaque stratégie. Cela permet de rapidement situer l'efficacité de la stratégie déterminée par l'algorithme DQN.

La cinquième est spécifique à l'algorithme DQN et affiche la Q-value qu'il affecte à chaque action à chaque step.
