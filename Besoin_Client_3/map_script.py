import folium
from folium.plugins import MarkerCluster
import pandas as pd
import branca.colormap as cm


def convert_csv_to_json(out_path, csv_path="csv_correction.csv"):
    write_json(pd.read_csv(csv_path), out_path)


def import_json(path: str):
    return pd.read_json(path)


def write_json(df, path):
    df.to_json(path)


def all_predict(pkl_path, json_path):
    """
    
    retourne le dataframe avec les predictions de chaque ligne
    contenu du json : dictionnaire {preparation_function, model}


    :param pkl_path: model and parameters
    :param json_path: dataset in json
    :return:
    """
    import pickle
    import pandas as pd
    # loading pkl file
    with open(pkl_path, 'rb') as f:
        pkl_dict = pickle.load(f)

    prep_data_dict = pkl_dict["prep_data_dict"]
    model = pkl_dict["model"]

    # now we import the data
    data = import_json(json_path)
    original_data = data.copy(deep=True)




    # now we prepare the data
    categorical_data_cols = ['clc_quartier', 'clc_secteur', 'fk_stadedev', 'fk_port', 'fk_pied', 'fk_situation', 'fk_nomtech', 'villeca', 'feuillage']
    boolean_cols = ['fk_revetement', 'remarquable']


    from sklearn.preprocessing import OneHotEncoder

    # encode categorical data
    #print(prep_data_dict["encoders"].keys())
    for col, ohe in prep_data_dict["encoders"].items():
        #print(type(ohe))
        #print(col)
        ohetransform = ohe.transform(data[[col]])
        data = data.join(ohetransform)

    data.drop(prep_data_dict["encoders"].keys(), axis=1, inplace=True)

    #encode boolean data
    for col in boolean_cols:
        new_col = []
        for cell in data[col]:
            new_col.append(1 if cell == "Oui" else 0)
        data[col] = new_col

    # now the predictions
    X = data[prep_data_dict["top_features"]]
    predictions = model.predict(X)

    original_data["storm_predictions"] = predictions
    y_proba = model.predict_proba(X)
    original_data['Probability_False'] = y_proba[:, 0]
    original_data['Probability_True'] = y_proba[:, 1]
    return original_data



def get_wind_saint_quentin(d:str= None):
    """
    Renvoie les vitesses de vent pour Saint Quentin pour la date passee en argument.
    :param data: date au format yyyy-mm-dd, par defaut aujourd hui
    :d: date format yyyy-mm-dd
    :return: dict, keys : "date" la date, "gust" la vitesse des rafales, "mean" la vitesse moyenne
    """
    import requests
    import datetime as dt
    import pickle
    from keys import API_KEY

    date = dt.datetime.now().strftime("%Y-%m-%d") if d is None else d
    request = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/saint%20quentin/{date}/{date}?unitGroup=metric&elements=datetime%2Cwindgust%2Cwindspeedmean&key={API_KEY}&contentType=json"
    #print("Request : ", request)
    r = requests.get(request)
    #print(r.json())
    # with open('meteo1.json', 'wb') as f:
    #     pickle.dump(r.json(), f, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "date" : date,
        "gust" : r.json()["days"][0]['windgust'],
        "mean" : r.json()["days"][0]['windspeedmean']
    }




def make_map(json_path="file1.json", filename="index.html"): # unused
    df3 = import_json(json_path)

    my_map = folium.Map(location=[49.844535, 3.290589], zoom_start=13)
    colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow'], vmin=0, vmax=df3['haut_tot'].max())

    for i in range(len(df3)):
        folium.Circle(
            location=[df3.iloc[i]['latitude'], df3.iloc[i]['longitude']],
            radius=df3.iloc[i]['tronc_diam'] / 62.8,
            fill=True,
            color=colormap(df3.iloc[i]['haut_tot']),
            fill_opacity=0.2,
            popup=f'<div style="width:150px">Hauteur totale : {df3.iloc[i]["haut_tot"]}m<br>'
                  f'Hauteur tronc : {df3.iloc[i]["haut_tronc"]}m<br>'
                  f'Diamètre tronc : {df3.iloc[i]["tronc_diam"] / 3.14159:.2f} cm<br>'
                  f'Age estimé : {df3.iloc[i]["age_estim"]} ans</div>'
        ).add_to(my_map)
    my_map.add_child(colormap)
    my_map.save(filename)

def add_essouche_bool_column(data, date:str=None, fake_wind_speed = None):
    """
    Retourne le dataframe data (apres predict) en lui ajoutant une colonne booléenne indiquant s'il faut marquer l'arbre comme
    en danger en fonction des données meteo fournies.

    appliquer un coef entre le Probability_True et la vitesse windgust
    :param data:
    :param date:
    :return:
    """

    wind_gust_speed = get_wind_saint_quentin(date)["gust"] if fake_wind_speed is None else fake_wind_speed
    print(fake_wind_speed)
    print(wind_gust_speed)

    # paliers echelle de Beaufort :
    # < 60km/h rien
    # 62 a 74       --> aficher >0.9
    # 75 a 88       --> afficher >0.8
    # 89 a 102      --> afficher >0.7 arbres parfois déracinés
    # 103 et plus   --> afficher >0.6

    if wind_gust_speed > 103:
        treshold = 0.6
    elif wind_gust_speed > 89:
        treshold = 0.7
    elif wind_gust_speed > 75:
        treshold = 0.8
    elif wind_gust_speed > 62:
        treshold = 0.9
    else:
        treshold = 1


    is_displayed = []  # array of bool
    #print(f'TRESHOLD : {treshold}')
    for i in range(len(data)):
        #print(f'PTrue : {data.iloc[i]["Probability_True"]:.2f} ; T : {treshold}')
        #print(data.iloc[i]['Probability_True'] > treshold)
        is_displayed.append(True) if (data.iloc[i]['Probability_True'] > treshold) else is_displayed.append(False)

    #print(data.shape)
    #print(len(is_displayed))
    data["is_displayed"] = is_displayed
    # va devoir afficher les arbres dont Probability_True est superieur a treshold
    #print(data['is_displayed'].value_counts())
    return data

def make_storm_map(df3, out_path, gradient_colors = True, weather:bool = True, date:str = None, fake_wind_speed = None):
    if weather:
        df3 = add_essouche_bool_column(df3, date, fake_wind_speed=fake_wind_speed)
    #print(df3['is_displayed'].value_counts())
    #print(df3.head())
    my_map = folium.Map(location=[49.844535, 3.290589], zoom_start=13)
    colormap = cm.LinearColormap(colors=['green', 'red'], vmin=0, vmax=1)

    for i in range(len(df3)):
        #print(f'W : {weather} ; D : {df3.iloc[i]["is_displayed"]}')
        if (weather and bool(df3.iloc[i]['is_displayed'])):
            folium.Circle(
                location=[df3.iloc[i]['latitude'], df3.iloc[i]['longitude']],
                radius=df3.iloc[i]['tronc_diam'] / 62.8,
                fill=True,
                color=colormap(df3.iloc[i]['Probability_True']) if gradient_colors else colormap(df3.iloc[i]['storm_predictions']),
                fill_opacity=0.2,
                popup=f'<div style="width:150px">Hauteur totale : {df3.iloc[i]["haut_tot"]}m<br>'
                      f'Hauteur tronc : {df3.iloc[i]["haut_tronc"]}m<br>'
                      f'Diamètre tronc : {df3.iloc[i]["tronc_diam"] / 3.14159:.2f} cm<br>'
                      f'Age estimé : {df3.iloc[i]["age_estim"]} ans<br>'
                      f'Proba False : {df3.iloc[i]["Probability_False"]:.8f}<br>'
                      f'Proba True : {df3.iloc[i]["Probability_True"]:.8f}</div>'
            ).add_to(my_map)

        if not weather:
            folium.Circle(
                location=[df3.iloc[i]['latitude'], df3.iloc[i]['longitude']],
                radius=df3.iloc[i]['tronc_diam'] / 62.8,
                fill=True,
                color=colormap(df3.iloc[i]['Probability_True']) if gradient_colors else colormap(
                    df3.iloc[i]['storm_predictions']),
                fill_opacity=0.2,
                popup=f'<div style="width:150px">Hauteur totale : {df3.iloc[i]["haut_tot"]}m<br>'
                      f'Hauteur tronc : {df3.iloc[i]["haut_tronc"]}m<br>'
                      f'Diamètre tronc : {df3.iloc[i]["tronc_diam"] / 3.14159:.2f} cm<br>'
                      f'Age estimé : {df3.iloc[i]["age_estim"]} ans<br>'
                      f'Proba False : {df3.iloc[i]["Probability_False"]:.8f}<br>'
                      f'Proba True : {df3.iloc[i]["Probability_True"]:.8f}</div>'
            ).add_to(my_map)
        # TODO : a refactor
    my_map.add_child(colormap)
    my_map.save(out_path)


if __name__ == '__main__':
    print("Hello !")
    #make_map("csv_correction.json")
    """import pickle
    # loading pkl
    with open('mlp_model_1.pkl', 'rb') as f:
        model = pickle.load(f)

    data = pd.read_csv('csv_correction.csv')


    res_df = predict(model, data) # returns dataframe avec predictions"""



    make_storm_map(all_predict("rf_model_1.pkl", "csv_correction.json"), "storm_map_3.html", date="2010-02-28")



    #write_json(res_df, "res_df.json")
    print(f'finished !')
