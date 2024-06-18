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


def predict(model, original_data):
    """

    :param model_path: modele (a importer avec pickle en dehors avant la fonction)
    :param dataframe: dataframe non traité, dataframe pandas
    :return: dataframe avec une colonne en plus, la prediction
    """
    from sklearn import preprocessing
    import copy


    # encodage des colonnes non numériques necessaires au modele
    data = original_data.copy(deep=True)
    list_col = ['fk_stadedev', 'clc_quartier', 'clc_secteur', 'feuillage', 'fk_port', 'fk_situation', 'fk_nomtech',
                'fk_arb_etat']
    for col in list_col:
        encoder = preprocessing.LabelEncoder()
        encoded = encoder.fit_transform(data[col])
        data[col] = encoded


    # colonnes utilisees par le modele
    all_cols_list = ["haut_tot",
                     'haut_tronc',
                     'tronc_diam',
                     'fk_stadedev',
                     'age_estim',
                     'clc_quartier',
                     'clc_secteur',
                     'feuillage',
                     'fk_port',
                     'fk_situation',
                     'fk_nomtech']

    X = data.filter(all_cols_list)

    predictions = model.predict(X)
    original_data["storm_predictions"] = predictions

    y_proba = model.predict_proba(X)
    original_data['Probability_False'] = y_proba[:, 0]
    original_data['Probability_True'] = y_proba[:, 1]
    return original_data






def make_map(json_path="file1.json", filename="index.html"):
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


def make_storm_map(df3, out_path, gradient_colors = True):
    my_map = folium.Map(location=[49.844535, 3.290589], zoom_start=13)
    colormap = cm.LinearColormap(colors=['green', 'red'], vmin=0, vmax=1)

    for i in range(len(df3)):
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
    my_map.add_child(colormap)
    my_map.save(out_path)


if __name__ == '__main__':
    print("Hello !")
    #make_map("csv_correction.json")
    import pickle
    # loading pkl
    with open('mlp_model_1.pkl', 'rb') as f:
        model = pickle.load(f)

    data = pd.read_csv('csv_correction.csv')


    res_df = predict(model, data) # returns dataframe avec predictions

    make_storm_map(res_df, "storm_map.html")
    #write_json(res_df, "res_df.json")
    print(f'finished !')
