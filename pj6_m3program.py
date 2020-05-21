# -*- coding: utf8 -*-
from init_libraries import *

def ctrl_note(ref_file, test_file):
    # Chargement du fichier servant à entrainer notre modèle
    # df = pd.read_csv("DATA/notes.csv")
    df = ref_file
    # Chargement du fichier à Evaluer et Préformattage (acp) avant application Modèle
    ex = test_file
    ex1 = ex.drop(columns = 'id').values
    # Préparation Données pour modele Regression Logistique
    x = df.copy()
    x = df.drop(columns = 'is_genuine').values
    y = df['is_genuine'].values
    y.astype(int)
    
    # Centrage et Réduction
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)
    ex_scaled = std_scale.transform(ex1)
    
    # Calcul des composantes principales
    n_comp = 4                                      # choix du nombre de composantes à calculer
    pca = decomposition.PCA(n_components=n_comp)
    pca.fit(x_scaled)
    x_projected = pca.fit_transform(x_scaled)
    ex_projected = pca.fit_transform(ex_scaled)
    
    # Création du Modèle de regression Logistique sur notre Dataset
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(x_projected, y)
    
    # Traitement Prédictions sur fichier Test (example) et sortie dans un Dataframe
    e  = ex.count()
    if e["id"] > 0 : 
        y_pred = logreg.predict(ex_projected)
        ex["predict"] = y_pred
        tx_conf = logreg.predict_proba(ex_projected)
        i = tx_conf[:,0]
        ex["Probabilité d'authenticité(%)"] = np.round((100-i*100), 2)
        ex = ex.replace({'predict' : 1}, "Le billet est Faux")
        ex = ex.replace({'predict' : 0}, "Le billet est Vrai")
    else : 
        print("Fichier vide")

    print("Nombre de billets : ",e["id"])
    return ex