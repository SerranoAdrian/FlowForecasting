from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import os
from torch_geometric.data import Data



def process_link_loads_csv(file_path, delete_columns = ['Link', 'Line', 'Dir', 'Order', 'From NLC', 'From ASC', 'To NLC', 'To ASC'], key_columns = ['From Station', 'To Station']):
    """key_columns: primary key that will define the nodes"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.rstrip()
    # couple_stations = []
    # df = df.drop(columns=delete_columns)
    # df = convert_columns_to_float(df, len(key_columns))

    # Group by and sum the rest of the columns
    # df = df.groupby(by=key_columns, as_index=False).sum()
    # df = df[df['Total'] != 0]
    # print(df.shape)

    return df


def convert_columns_to_float(df, n):
    for col in df.columns[n:]:  # On ignore les premières colonnes jusqu'à n
        # Vérifier si la colonne n'est pas déjà de type float
        if not pd.api.types.is_float_dtype(df[col]):
            df[col] = (
                df[col]
                .astype(str)  # Convertir en str pour éviter les erreurs sur NaN
                .str.replace('\u202f', '', regex=True)  # Supprimer l’espace insécable
                .str.replace(',', '', regex=True)  # Supprimer les virgules
                .replace('nan', '0')  # Remplacer les 'nan' textuels
                .fillna(0)  # Remplacer les NaN réels
                .astype(float)  # Convertir en float
            )
    return df



def get_graph_attributes(folder_path):
    """
    return:
    - num_nodes: the number of unique link (inter-station)
    - edge_index: the list of nodes that are linked, e.g. (a,b) means from node a you can go to node b
    - node_mapping: dictionary that gives relation between the link (string) to node_id (int)
    - df_list: list of dfs obtained from processing csv and link columns, chronologically order !!!!!TO DO
    """
    link_set = set([])
    edge_list_set = set([])
    dfs = {}

    filenames = order_by_time(folder_path)
    
    #  Parcourir les csv
    for filename in filenames:
        # Process df
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.rstrip()
        df = df_correct_link_id(df)
        dfs.update({filename: df})

        # Collect unseen links
        link_set.update(df['Link'])

        # Collect unseen edges
        edge_list = df_to_edge_index(df)
        edge_list_set.update(edge_list)

    node_mapping = {link: node for node, link in enumerate(link_set)} # a mettre plus tard
    num_nodes = len(node_mapping)
    edge_list_set = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in edge_list_set]
    edge_index = torch.tensor(list(edge_list_set), dtype=torch.long).t()

    return num_nodes, edge_index, node_mapping, dfs


def order_by_time(folder_path):
    days = ["MON", "MTT", "TWT", "FRI", "SAT", "SUN"]
    years = range(16,24)
    ordered_file = []
    for year in years:
        for day in days:
            filename = f"Link_Loads_NBT{year}{day}.csv"
            file_path = os.path.join(folder_path, filename)
            #Use a try-except function so different naming conventions don't cause errors
            if os.path.isfile(file_path):
                #Add each file in NUMBAT as a dataframe to the dictionary
                ordered_file.append(filename)
    return ordered_file


def df_correct_link_id(df, column="Link"):
    # check_link_format(df, column="Link")
    corrected_links = []

    for link in df[column]:
        parts = link.replace('@', '_').split('_')
        line1 = parts[1]
        line2 = parts[3]
        expected_line = parts[5]
        if line1 != expected_line:
            parts[1] = expected_line
        if line2 != expected_line:
            parts[3] = expected_line
        
        reconstructed_link = "_".join(parts[:-1]) + "@" + parts[-1]
        corrected_links.append(reconstructed_link)
    
    df[column] = corrected_links
    return df


def df_to_edge_index(df):
    # Create the node list and corresponding mapping
    unique_nodes = list(set(list(df.index))) # Extraire tous les nœuds uniques dans edge_index
    node_mapping = {int(node): i for i, node in enumerate(unique_nodes)}  # Créer un mapping vers de nouveaux indices
    df.index = df.index.map(node_mapping)  # Réindexer les nœuds dans edge_index

    edge_list = []
    for end_link, end_station in zip(df['Link'], df['To NLC']): # NLC is the station ID, the station name can be different reagrding where the metro comes from
        links_connected = [start_link for start_link, start_station in zip(df['Link'], df['From NLC']) if start_station == end_station]
        for link_connected in links_connected:
            edge_list.append((end_link, link_connected))

    # for node, row in enumerate(df.iterrows()):
    #     end_station = row[1]['To Station']
    #     # Search for all the stations that 
    #     nodes_connected = list(df[df['From Station'] == end_station].index)
    #     for nc in nodes_connected:
    #         edge_list.append((node, nc))

    # edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return edge_list


def add_missing_nodes(df, node_mapping, num_nodes):
    df.index = [node_mapping[link] for link in df['Link']] # Change index as node_id
    df = df.iloc[:, 17:] # Keep only flows columns
    df = df.groupby(level=0).sum() # Make sure of the node unicty
    full_index = pd.Index(range(num_nodes), name="node_id")
    df = df.reindex(full_index, fill_value=0) # fill missing nodes values
    return df


def build_quarter_hour_data(df, filename, num_nodes):
    """
    return: 
    - df_qhrs: the list of 24*4 df for each quarter-hour (encode temporal features and the flow)
    """

    df_qhrs = [df.iloc[:,i] for i in range(df.shape[1])]
    day = filename.split('.')[0][-3:]
    if day not in ['FRI','SAT','SUN']:
        day = 'MTWT'
    df_one_hot_encoding_day = pd.DataFrame(np.zeros((num_nodes, 4)), columns=['MTWT', 'FRI', 'SAT','SUN'])
    df_one_hot_encoding_day[day] = 1.0
    df_qhrs = [pd.concat([df, df_one_hot_encoding_day], axis=1) for df in df_qhrs]
    
    for i in range(len(df_qhrs)):
        df_qhrs[i].columns.values[0] = 'Flow'
        day_time = get_day_time(i)
        df_one_hot_encoding_day_time = pd.DataFrame(np.zeros((num_nodes, 6)), columns=['Early', 'AM Peak', 'Midday','PM Peak', 'Evening', 'Late'])
        df_one_hot_encoding_day_time[day_time] = 1.0
        df_qhrs[i] = pd.concat([df_qhrs[i], df_one_hot_encoding_day_time], axis=1)

    return df_qhrs


def get_day_time(i):
    if i < 8: # 0500-0700
        return 'Early'
    elif i < 20: # 0700-1000
        return 'AM Peak'
    elif i < 36: # 1000-1600
        return 'Midday'
    elif i < 48: # 1600-1900
        return 'PM Peak'
    elif i < 60: # 1900-2200
        return 'Evening'
    else: # 2200-0500
        return 'Late'
    

def df_to_graph(df, edge_index):
    x = torch.tensor(df[['MTWT','FRI', 'SAT','SUN','Early', 'AM Peak', 'Midday','PM Peak', 'Evening', 'Late']].values, dtype=torch.float32)
    edge_index = edge_index
    y = torch.tensor(df[["Flow"]].values, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y)



##### Not Important #####


def check_link_format(df, column="Link"):
    # Définition du pattern regex pour accepter MAJUSCULES et minuscules
    pattern = r"^[A-Za-z]{4}_[A-Za-z]{3}_[A-Za-z]{2}>[A-Za-z]{4}_[A-Za-z]{3}_[A-Za-z]{2}@[A-Za-z]{3}$"
    
    # Filtrer les valeurs qui ne correspondent pas au pattern
    invalid_links = df[~df[column].str.match(pattern, na=False)]  # na=False pour ignorer les NaN
    
    # Afficher les valeurs invalides
    if not invalid_links.empty:
        print("Valeurs invalides trouvées :")
        print(invalid_links[column].tolist())
    else:
        print("Toutes les valeurs sont bien formatées !")
    
    return invalid_links


def process_NBT_outputs_csv2(file_path):
    # Reading the CSV file
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = df.columns.str.rstrip()



def process_NBT_outputs_csv(file_path):
    # Reading the CSV file
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = df.columns.str.rstrip()
    
    name_mistakes = {
            'Canning Town (Stratford)': 'Canning Town',
            'Canary Wharf DLR (Stratford)': 'Canary Wharf DLR',
            'Liverpool Street NR (Chingford)': 'Liverpool Street NR',
            'Hackney Downs (Enfield / Chesthunt)': 'Hackney Downs',
            'Hackney Downs (Chingford Branch)' : 'Hackney Downs',
            'Kennington (Charing Cross)': 'Kennington',
            'Kennington (Bank)': 'Kennington',
            'Camden Town (High Barnet)': 'Camden Town',
            'Camden Town (Edgware)': 'Camden Town',
    }

    # Creating a dictionary where each key (NLC) has a list of associated stations
    nlc_to_stations = defaultdict(list)
    
    proccesed_station_names = [station if station not in name_mistakes else name_mistakes[station] for station in df['From Station']]
    df['From Station'] = proccesed_station_names
    df['To Station'] = [station if station not in name_mistakes else name_mistakes[station] for station in df['To Station']]

    # Filling the dictionary with values from the DataFrame
    for nlc, station in zip(df['From NLC'], proccesed_station_names):
        nlc_to_stations[nlc].append(station)

    # Displaying NLC codes with multiple stations
    for nlc, stations in nlc_to_stations.items():
        if len(set(stations)) > 1:  # Only keep those with multiple associated stations
            print(f"!!! The following NLC is associated with multiple station names !!!  {nlc} -> {set(stations)}")
        nlc_to_stations[nlc] = list(set(stations))[0]
        
    nlc_to_stations = dict(nlc_to_stations)
    print(f"Number of stations: {len(nlc_to_stations)}")
    return df

def get_couple_stations(df):
    couple_stations = []
    df = df.drop(columns=['Link', 'Line', 'Dir', 'Order', 'From NLC', 'From ASC', 'To NLC', 'To ASC'])
    # Group by and sum the rest of the columns
    df = df.groupby(['From Station', 'To Station']).sum()
    return df
    # for nlc, station in zip(df['From Station'], df['To Station']):
    #     if station in nlc_to_stations:
    #         couple_stations.append((station, nlc_to_stations[station]))
    # return couple_stations