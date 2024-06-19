import pandas as pd
import json
import pickle

def age_estim(data_json, dico):
    df = pd.DataFrame(json.loads(data_json))
    X = df[['haut_tronc', 'tronc_diam', 'fk_stadedev', 'clc_nbr_diag', 'fk_nomtech', 'haut_tot']]

    # Update the dictionary keys according to your setup
    X['fk_stadedev'] = dico["encoder"].fit_transform(X[['fk_stadedev']])
    X['fk_nomtech'] = dico["encoderlabel"].fit_transform(X[['fk_nomtech']])
    X = dico["scaler_feature"].fit_transform(X)

    pred = dico["RandomForest"].predict(X)
    pred = pred.reshape(-1, 1)
    pred = dico["scaler_age"].inverse_transform(pred)
    
    age_estimated = pd.DataFrame(pred, columns=['age_estim'])
    
    # Convert to JSON and write to a file
    age_estimated_json = age_estimated.to_json(orient='records')
    with open("Besoin_Client_2/age_estim.json", "w") as f:
        f.write(age_estimated_json)
    
    return age_estimated_json

# Load the pickle dictionary
with open("Besoin_Client_2/cornichon.pkl", "rb") as f:
    dico = pickle.load(f)
    print(dico.keys())

# Load the JSON data
with open("Besoin_Client_2/data_test.json", "r") as f:
    data_json = f.read()

# Call the function
age_estim(data_json, dico)






