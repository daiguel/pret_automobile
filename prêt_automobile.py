import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.utils import resample
from sklearn.utils import shuffle
from keras.models import Sequential  #initialise le resaux de neurones 
from keras.layers import Dense,Dropout # module pour créer les couches de resaux de neurones
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def import_data():
    # lire la base à partir du fichier csv
    banque = pd.read_csv('base_banque.csv', sep=";", encoding='UTF-8')

    # verification de dataset si contient des valeurs manquantes
    nan_verifiation = banque.isnull().values.any()
    print(nan_verifiation)#resultat: False

    
    ###########rééchantillonnage
    """ 
    train, test = train_test_split(banque, test_size=0.3, random_state=1)
    X_train_0 = train[train['P_P_Auto']==0]
    X_train_1 = train[train['P_P_Auto']==1]
    X_train_1 = resample(X_train_1, 
                     replace=True,
                     n_samples=X_train_0.shape[0],
                     random_state=1)
    X_train_1 = X_train_1.reset_index()
    banque = pd.concat([X_train_0, X_train_1])
    banque = shuffle(banque, random_state=1)
    """
    #########
    # extraction des variables dépendentes et indépendentes
    y = banque.P_P_Auto
    # enlever le numéro du client qui ne sert a rien
    x = banque.drop(["P_P_Auto", "Numero_Client"], axis=1)
    return x, y



def variable_categorique(x):
    """cette fonction permet de convertir les variables categorique en dummies variable"""
    # traitement des variables categorique
    x_object = x.iloc[:, :].values

    # I_Salarie_CR 3
    dummies = pd.get_dummies(pd.Series(x_object[:, 2]))
    # suppresion d'une variable dans notre cas Emp
    dummies = dummies.drop(["Emp"], axis=1)

    # Nouveau_Client 6
    labelEncoder_2 = LabelEncoder()
    dummies_2 = labelEncoder_2.fit_transform(pd.Series(x_object[:, 5]))
    dummies_2 = pd.DataFrame(dummies_2, columns=["Nouveau_Client"])

    # Metier_portef 13
    dummies_3 = pd.get_dummies(pd.Series(x_object[:, 12]))
    # suppresion d'une variable, dans notre cas 0000

    dummies_3 = dummies_3.drop(["0000"], axis=1)
    # concatener les differents dataframe

    # regrouper les variable categorique dans une seul dataframe
    x_var_cat = pd.concat([dummies, dummies_2, dummies_3], axis=1)

    # suppression des colonnes de type String dans la dataframe
    x = x.drop(["Nouveau_Client", "Metier_portef", "I_Salarie_CR"], axis=1)

    # concat variables categorique codifier
    x = pd.concat([x, x_var_cat], axis=1)

    return x



def classe_des_variables(banque):
    #en utilisant un filtre sur excel on a pu mettre dans un fichier csv contenant les noms des variables quantitatives
    var_quanti=pd.read_csv("variables_quantitatives.csv",sep=";",encoding='UTF-8')

    #convertire le type de variables quantitatives de dataframe a liste
    list_var_quanti= var_quanti.iloc[:, 0].tolist()

    #on va extraire toute la premiere column de notre base banque pour trouver aussi toutes les variable qualitatives
    list_var=list(banque.columns)

    #chercher les varriables qualitatives
    list_var_quali=[]
    for i in list_var:
        if i not in list_var_quanti:
            list_var_quali.append(i)
    #renvoi des deux listes varibales qualitatives et variables qualitatives
    return  list_var_quanti, list_var_quali



def supp_var_corr(banque,list_var_quanti):
    """cette fonction permet de supprimer les varaiable quantitaf qui ont des correlations"""
    # construire un dataframe avec seulement les variables quantitaves de notre base,
    # NB x_quanti est juste une liste qui contient les noms des variables quantitaves
    banque_var_quanti = banque[list_var_quanti]

    #tableau de corrélation
    tab_corr = banque_var_quanti.corr()

    # chercher les correlations entre les variables de sorte qu'on forme une liste qui
    # aura un tuple pour chaque deux variables ou leur corrélation dépasse 0.7
    list_var_corr = []
    for i in range(len(list_var_quanti)):
        for j in range(i + 1, len(list_var_quanti)):
            if (tab_corr.iloc[i, j] >= 0.75):
                list_var_corr.append((list_var_quanti[i], list_var_quanti[j]))

    #39 corrélations trouvé

    #liste des variables quantitaves à supprimer
    list_col_supp = ['Tarif_reel', 'Age_PP', 'N_enfants_CC', 'N_Adultes_CC', 'Dispo_Mensuel', 'Flux_Corr', 'T_Collecte',
                     'T_Ep_Taux_Fixe', 'T_Epar_Assur', 'T_Epargne', 'E_DAV', 'E_CSL', 'T_PEA', 'T_Cpt_Titres',
                     'T_P_Habitat', 'T_P_Familiaux', 'T_Carte', 'T_Contrats_IARD', 'T_IVP_VP', 'N_Ope_debit',
                     'N_Ope_Tot', 'N_Paiem_Carte', 'N_Ope_credit', 'N_Livrets', 'N_Cpt_Tit_PEA', 'N_Prod_Collecte']
    # suppression des variables corrélés
    banque = banque.drop(columns=list_col_supp)

    return banque



#stepwise_selection ou  backward elimination
def backword_selection(X, y, threshold_out=0.05,verbose=True):
    """cette fonction renvois les variables a eleiminé du dataset"""
    changed = True
    excluded=[]#initialisation de la liste des variables a enlevé
    while changed:

        # les étapes de backward
        #entrainé le modele pour recuperer la p-value
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X))).fit()
        #trouver toutes les pvalues de toutes les variables
        pvals = model.pvalues.iloc[1:]
        #recherche la pvalue la plus grande(ou la la plus mauvaise)
        worst_pval = pvals.max()
        if worst_pval > threshold_out:
            #trouver la variable concerene par la plus grande pvalue
            worst_feature = pvals.idxmax()
            # enlever la variable concerné pour avoir un modéle plus pérformant
            X=X.drop(worst_feature,axis=1)
            #costruction de la liste des variables a supprimer
            excluded.append(worst_feature)
            #verbos a True signifie afficher ce messaga dans la console si non ne rien faire
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        else:#une fois les pvalues presentes sont au-dessous de threshold_out on s'arréte
            changed=False
    return excluded



#forward selection 
def forward_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # les étapes de forward 
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break
    return included



def divise_data(x,y):
    #devise la table une partie pour trainning et l'autre pour test 
    X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    
    #Changer l'echelle des varaibles
    sc = StandardScaler() 
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    
    return X_train,X_test,Y_train,Y_test



def model ():
    #Construction d'un réseau de neuronne 
    #initialisation 
    classifier = Sequential()
    #ajouter la couche d'entree et une couche caché 
    classifier.add(Dense(units=27, activation="relu",kernel_initializer="uniform",input_dim=54))
   
    #Ajouter une deuxième couche cachée 
    classifier.add(Dense(units=27, activation="relu",kernel_initializer="uniform"))
   
    #Ajouter la couche de sortir 
    classifier.add(Dense(units=1, activation="sigmoid",kernel_initializer="uniform"))
    
    classifier.compile(optimizer="adam" ,loss="binary_crossentropy" ,metrics=["accuracy"]) 
    
    
    return classifier



def models(X_train,Y_train,n_Y):
    kfold = KFold(n_splits=10, random_state=1)
    modeles = [('Régression logistique', LogisticRegression()), 
           
           
           ('Analyse discriminante', LinearDiscriminantAnalysis()),
           ('Random Forest', RandomForestClassifier())]
    for elt in modeles:
        results = cross_val_score(elt[1], X_train, Y_train, 
                                  cv=kfold, scoring='accuracy')
        print(elt[0], ' - score : ', results.mean())
        mod = elt[1]
        Y_ = mod.fit(X_train, Y_train).predict(X_test)
        table = pd.crosstab(Y_test, Y_)
        print("erreur de prédiction sur variable cible, sur l'échantillon test :", 
              table[0][1] / n_Y)

##########################################################
#main program

# préparation de données
x,y=import_data()

#cette fonction est necessaire pour l'execution de backward Elimination
x=variable_categorique(x)

#trouver les variables qualitatives, et les variables quantitaves
x_quanti,x_quali=classe_des_variables(x)

#suppression des varibales corrélés
x=supp_var_corr(x,x_quanti)

col=list(x.columns)

#chercher les  variables a eliminer avec backword 
result = backword_selection(x, y)#62 variables a eliminersur sur 116
#result = backword_selection(x, y)
liste_x_reduit = list(set(x.columns) - set(result))#54 variables restentes

#extraction du dataset avec les variable 
x_reduit=x[liste_x_reduit]

#diviser les donneés en apprentissage et test 
X_train,X_test,Y_train,Y_test=divise_data(x_reduit,y)

# appel pour consrtuire un  model de réseaux de neuronnes 
modele_R_N=model()
#Entrainer le reseau de neurones
modele_R_N.fit(X_train,Y_train,batch_size=10,epochs=10)
#proba d'avoir un prêt
y_pred = modele_R_N.predict(X_test)
#calcul de l'AUC
print('Réseaux de neuronnes AUC :', roc_auc_score(Y_test, y_pred))
y_pred = (y_pred>0.5)
#matrice de conffusion
cm = confusion_matrix (Y_test,y_pred)
#pourcentage de bonne prédection
print ('Réseaux de neuronnes Accuracy',metrics.accuracy_score(Y_test,y_pred))

#appel au model de regression logistic 
#pas besoin de definir les types as category car on a encoder les variables
modele_L_R=LogisticRegression()
#Entrainer la regression logitique
modele_L_R.fit(X_train,Y_train)
#predire sur le test 
y_pred1 =modele_L_R.predict(X_test)  
y_pred1 = (y_pred1>0.5)
#proba
y_proba1=modele_L_R.predict_proba(X_test)
#matrice de conffusion
cm1=confusion_matrix(Y_test,y_pred1)
#calcul de l'AUC
print('Regression Logistique AUC :', roc_auc_score(Y_test, y_proba1[:,1]))
#pourcentage de bonne prédection
print ('Regression Logistique Accuracy',metrics.accuracy_score(Y_test,y_pred1))


#random forest
modele_R_F_C=RandomForestClassifier()
#Entrainer le randome forrest classifier 
modele_R_F_C.fit(X_train,Y_train)
#predire sur le test 
y_pred2=modele_R_F_C.predict(X_test)
#matrice de conffusion
cm2 = pd.crosstab(Y_test, y_pred2)
#proba
y_proba2=modele_R_F_C.predict_proba(X_test)
#calcul de l'AUC
print('Random Forest Classifier AUC :', roc_auc_score(Y_test,y_proba2[:,1]))
#pourcentage de bonne prédection
print ('Random Forest Classifier Accuracy',metrics.accuracy_score(Y_test,y_pred2))



#applique la méthodes de cross validation pour bien determiner le modele le plus performant  
#kfold a vec k=10 cette etape est necessaire pour que l'evalution sur les défferentes modele sois sur les mémes échantillons
kfold = KFold(n_splits=10, random_state=1)#une seul separation pour tout les modéles

#Réseaux de neuronnes @ca prend un peu de temps pour finir l'exucution
#evaluation de l'accuracy
classifier_R_N = KerasClassifier(build_fn=model,batch_size=10,epochs=10)
accuracy_R_N = cross_val_score(classifier_R_N,X=X_train,y=Y_train, cv=kfold,scoring='accuracy')
#pourcentage de bonne prédection
print("Réseaux de neuronnes Accuracy Kfold ",accuracy_R_N.mean())

AUC_R_N = cross_val_score(classifier_R_N,X=X_train,y=Y_train, cv=kfold,scoring='roc_auc')
#chercher la moyenne des AUC
print("Réseaux de neuronnes AUC Kfold ",AUC_R_N.mean())



#Regression Logistique 
#evaluation de l'accuracy
classifier_L_R = LogisticRegression()
accuracy_L_R = cross_val_score(classifier_L_R,X=X_train,y=Y_train, cv=kfold,scoring='accuracy')
#pourcentage de bonne prédection 
print("Regression Logistique  Accuracy Kfold ",accuracy_L_R.mean())

#evaluation de l'AUC
AUC_L_R = cross_val_score(classifier_L_R,X=X_train,y=Y_train, cv=kfold,scoring='roc_auc')
#chercher la moyenne des AUC
print("Regression Logistique  AUC Kfold ",AUC_L_R.mean())



#Random Forest Classifier 
#evaluation de l'accuracy
classifier_R_F_C = RandomForestClassifier()
accuracy_R_F_C = cross_val_score(classifier_R_F_C,X=X_train,y=Y_train, cv=kfold,scoring='accuracy')
#pourcentage de bonne prédection 
print("Random Forest Classifier  Accuracy Kfold ",accuracy_R_F_C.mean())

#evaluation de l'AUC
AUC_R_F_C = cross_val_score(classifier_L_R,X=X_train,y=Y_train, cv=kfold,scoring='roc_auc')
#chercher la moyenne des AUC
print("Random Forest Classifier  AUC Kfold ",AUC_R_F_C.mean())

#random forest classifier est donc le meilleure

#on cherche a exclure les individus qui ont bénificier d'un prêt 
banque= pd.concat([x_reduit,y],axis=1)
#pour pouvoir retrouvé les clients a la fin vue qu'on va supprimer ceux qui ont déja un prêt Automobile 
datframebanque=pd.read_csv('base_banque.csv', sep=";", encoding='UTF-8')
Numero_Client=datframebanque.Numero_Client
banque= pd.concat([Numero_Client,banque],axis=1)
idivdus_sans_pret = banque [banque['P_P_Auto']==0]
#garder l'indexation des clients pour pouvoir les retrouvés
index_cl_sans_pret=idivdus_sans_pret.Numero_Client
#☺supprimer P_P_Auto on aura pas besoin
idivdus_sans_pret=idivdus_sans_pret.drop(["P_P_Auto", "Numero_Client"], axis=1)

#random forest classifier 
#trouver les proba davoir un pret pour tout les client qui ne l'ont pas encore eu
proba_R_F_C = model_R_F_C.predict_proba(idivdus_sans_pret)
#concatener chaque client avec ca probabilité d'avoir un prêt 
rslt_R_F_C=pd.concat([pd.Series(proba_R_F_C[:,1]),index_cl_sans_pret],axis=1)
#trié le dataframe avec la probaiilite la plus grande a la plus petite
rslt_R_F_C=rslt_R_F_C.sort_values(0, ascending=False)
#extraire les 1000  les personne les plus apétentes au prêt 
rslt_R_F_C=rslt_R_F_C.head(10000)#le résultats final est stocké dans cette variables le numéro des 10000 clients les plus appétentes



#la section en bas est un plus c'est juste pour etudier la différence du resultat a la fin entre chaque modele

#Réseau de neuronnes
#trouver les proba davoir un pret pour tout les client qui ne l'ont pas encore eu  
proba_R_N = modele_R_N.predict(idivdus_sans_pret)
#concatener chaque client avec ca probabilité d'avoir un prêt 
index_cl_sans_pret=index_cl_sans_pret.reset_index(drop=True)
rslt_R_N=pd.concat([pd.Series(proba_R_N[:,0]),index_cl_sans_pret],axis=1)
#trié le dataframe avec la probaiilite la plus grande a la plus petite
rslt_R_N=rslt_R_N.sort_values(0, ascending=False)
#extraire les 1000  les personne les plus apétentes au prêt 
rslt_R_N=rslt_R_N.head(10000)

#regression loguistique
#trouver les proba davoir un pret pour tout les client qui ne l'ont pas encore eu
proba_L_R = modele_L_R.predict_proba(idivdus_sans_pret)
#concatener chaque client avec ca probabilité d'avoir un prêt 
index_cl_sans_pret=index_cl_sans_pret.reset_index(drop=True)
rslt_L_R=pd.concat([pd.Series(proba_L_R[:,1]),index_cl_sans_pret],axis=1)
#trié le dataframe avec la probaiilite la plus grande a la plus petite
rslt_L_R=rslt_L_R.sort_values(0, ascending=False)
#extraire les 1000  les personne les plus apétentes au prêt 
rslt_L_R=rslt_L_R.head(10000)




#compter le nombres d'inidividus défférents R_F_C vs L_R
liste_defferenceR_F_C_vs_L_R=[]
for i in rslt_R_F_C.Numero_Client:
    if i not in rslt_L_R.Numero_Client:
        liste_defferenceR_F_C_vs_L_R.append(i)
        
##compter le nombres d'inidividus défférents R_F_C vs R_N
defferenceR_F_C_vs_L_R=[]
for i in rslt_R_F_C.Numero_Client:
    if i not in rslt_R_N.Numero_Client:
        liste_defferenceR_F_C_vs_R_N.append(i)
        
##compter le nombres d'inidividus défférents L_R vs R_N
liste_defferenceL_R_vs_R_N=[]
for i in rslt_L_R.Numero_Client:
    if i not in rslt_R_N.Numero_Client:
        liste_defferenceL_R_vs_R_N.append(i)
