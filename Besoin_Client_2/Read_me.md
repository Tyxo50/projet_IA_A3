# Fonctionnement du fichier script.py

La fonction présente dans le script fonctionne comme ceci :

    - Elle prend en entrée un fichiers JSON contenant un jeu de donnée à tester ainsi qu'un dictionnaire (fichiers .pkl recuperer en dehors de la fonction)
    - Elle encode et scale les colonne suivante 'haut_tronc','tronc_diam','fk_stadedev','clc_nbr_diag','fk_nomtech','haut_tot'
    - Elle réalise ensuite une prédiction avec le modèle RandomForest (déjà entrainer ultérieurement)
    - Elle creer ensuite un fichiers JSON age_estim avec la prédiction de tout les âges

Pour executé le script il suffit de se munir des fichiers suivants (cornichon.pkl et data_test.json) et de taper la commande suivante :
        `python script.py`

Le fichier age_estim.json sera automatiquement créer.
