#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf8 -*-
import time   # Librairie temps pour calculs durée par exemple
trt_start_time = time.time()

from pj6_m3program import *

get_ipython().run_line_magic('cd', 'DATA')


# In[2]:


try:
    # Chargement du fichier à Evaluer et Préformattage (acp) avant application Modèle
    print("\n" * 2)
    print("Veuillez renseigner le nom du fichier 'Reference' initial qui sera utilisé pour entrainer le modèle de Régression Logistique.")
    print("Indiquez son chemin éventuel s'il n'est pas dans le répertoire 'en cours' et SANS l'extension(csv)).")
    print("Si blanc, le fichier par defaut sera : DATA/notes.csv")
    ref_file = input("Fichier Reference =  ")
    if ref_file == "":
        ref_file = "notes"
    print("")
    ref = pd.read_csv(ref_file +".csv")
    
    try:
        # Chargement du fichier à Evaluer et Préformattage (acp) avant application Modèle
        test_file = input("Veuillez entrer le nom de votre fichier csv (SANS l'extension) : ")
        print("")
        print("Si blanc, le fichier de test utilisé sera : DATA/test_model.csv")
        if test_file == "":
            test_file = "test_model"
        print("")
        ex = pd.read_csv(test_file +".csv")
        ctrl_note(ref, ex)
        ex.to_excel(test_file+"_RESULTAT_ANALYSE.xlsx")
        display(ex)
       
    except FileNotFoundError:
        print("")
        print(120*'/')
        print("")
        print("   LE FICHIER A TESTER ", test_file +".csv", "  N'AS PAS ETE TROUVE OU EST INCORRECT !!!")
        print("")
        print(120*'/') # print(*50*('/',), sep='_')
    
except FileNotFoundError:
    print("")
    print(120*'/') # print(*50*('/',), sep='_')
    print("")
    print("   LE FICHIER DE REFERENCE ", ref_file +".csv", "  N'AS PAS ETE TROUVE OU EST INCORRECT !!!")
    print("")
    print(120*'/')
    
dureetotale = round(time.time() - trt_start_time, 5)
print("--- Durée TOTALE du Notebook PJ6 Test Model --- ", "%s seconds" % dureetotale)

