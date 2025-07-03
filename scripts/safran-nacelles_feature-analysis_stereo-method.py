#%%
# --- Imports standards ---
import os
import pickle
from pathlib import Path

# --- Imports scientifiques ---
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, chi2
from scipy.spatial import ConvexHull

# --- Machine Learning / Préprocessing ---
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Visualisation ---
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.path import Path as MplPath  # évite conflit avec pathlib.Path
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import Counter,defaultdict

# --- Traitement géométrique ---
from shapely.geometry import Polygon
from shapely.ops import unary_union


# === Paramètres ===
path = '/media/rbeauvais/Elements/romainb/2025-n05-Safran-nacelles-FoD/'
date = '2025-04-01'
channel = 2
sensor = 'zoom-f4'
sensor_id = ''
feature_list = [ 'crest_factor', 'ultrasoundlevel','spectralflux']
features_to_display = 'all'
background_label = 'rotation'
chi_percent = 99
x_thresh = 20
y_thresh = 10


# Paramètres
machine_name = ''
batch = 'test'  # 'train' Ou 'test'
histogram_n_bits = 6
n_clusters = 50
threshold = 10 # (%)

rolling_window = 0.5 # (s)

class EllipseCluster:
    def __init__(self, centroid, width, height, angle):
        self.centroid = centroid  # Centre de l'ellipse
        self.width = width        # Largeur (2 * demi-grand axe)
        self.height = height      # Hauteur (2 * demi-petit axe)
        self.angle = angle        # Angle d'inclinaison en degrés

    # Méthode pour créer l'ellipse
    def create_ellipse(self):
        return Ellipse(xy=(self.centroid[0], self.centroid[1]), 
                       width=self.width, 
                       height=self.height, 
                       angle=self.angle, 
                       edgecolor='blue', 
                       fc='None', 
                       lw=2)

    def get_centroid(self):
        """Récupérer le centroid de l'ellipse."""
        return self.centroid

# Fonction pour sauvegarder les percentiles dans un fichier CSV
def save_percentiles_to_csv(percentiles,machine_name, filepath='percentiles.csv'):
    df = pd.DataFrame(percentiles).T
    df.columns = ['1st_percentile','5th_percentile', '95th_percentile','99th_percentile']
    df.index.name = 'feature'
    os.makedirs(f'results/{machine_name}', exist_ok=True)
    df.to_csv(f'results/{machine_name}/{filepath}')


def display_anomaly_detector(results,filtered,labels_pca,label_colors):
    
    # =========================================
    # === Attribution des couleurs par label ==
    # =========================================
    label_colors['other'] = 'rgba(0,0,0,1)'  # noir opaque
    df_binary = pd.DataFrame(results['Label']).copy()  # Conserve uniquement la colonne 'Label'
    df_binary['binary'] = results['Label'].isin(['cutter', 'rivets',"rivets (pas d'impact)"]).astype(int)

    # Masque des anomalies detéctées par la PCA analysis
    anomaly_times = filtered.index[labels_pca == 'Anomalie']
    anomalie_values = df_binary.loc[anomaly_times]

    # 3. Calculer les cas (positifs = anomalies détectées, négatifs = non détectées)
    # TP : anomalie détectée et réellement un impact (binary == 1)
    vp = ((anomalie_values['binary'] == 1)).sum()

    # FP : anomalie détectée mais pas d’impact réel (binary == 0)
    fp = ((anomalie_values['binary'] == 0)).sum()

    # VN : pas d’anomalie détectée et pas d’impact réel
    non_anomaly_times = df_binary.index.difference(anomaly_times)
    non_anomaly_values = df_binary.loc[non_anomaly_times]
    vn = ((non_anomaly_values['binary'] == 0)).sum()

    # FN : pas d’anomalie détectée mais il y avait un impact réel
    fn = ((non_anomaly_values['binary'] == 1)).sum()

    binary = df_binary['binary'].values
    index = df_binary.index

    # Détection des fronts montants et descendants
    rising_edges = np.where((binary[:-1] == 0) & (binary[1:] == 1))[0] + 1  # +1 pour pointer sur le 1
    falling_edges = np.where((binary[:-1] == 1) & (binary[1:] == 0))[0] + 1

    # Associer chaque front montant à son front descendant suivant
    events = []
    j = 0
    for start in rising_edges:
        while j < len(falling_edges) and falling_edges[j] <= start:
            j += 1
        if j < len(falling_edges):
            end = falling_edges[j]
            labels_in_event = df_binary.iloc[start:end+1]['Label'].unique().tolist()
            events.append({
                'start': start,
                'end': end,
                'labels': labels_in_event
            })


    # Initialiser un compteur
    label_counter = Counter()

    # Compter chaque label dans chaque événement
    for event in events:
        label_counter.update(event['labels'])

    # Initialiser un compteur par label
    detected_by_anomaly = defaultdict(int)

    # Vérifier les anomalies dans chaque segment
    anomaly_idx_set = set(anomaly_times)  # plus rapide pour les recherches
    has_anomaly_per_event = []

    for event in events:
        start = event['start']
        end = event['end']
        label = event['labels'][0]  # on suppose que 'label' est inclus dans chaque event
        time_range = index[start:end]
        has_anomaly = any(t in anomaly_idx_set for t in time_range)
        has_anomaly_per_event.append(has_anomaly)
        
        if has_anomaly:
            detected_by_anomaly[label] += 1

    for label, count in label_counter.items():
        print(f"{label} : {count} événements")
        print(f"  → {detected_by_anomaly[label]} détectés comme anomalies")

    print(f"{sum(has_anomaly_per_event)} événements contiennent au moins une anomalie sur {len(events)}")
    print(f"Vrais positifs (TP): {vp}")
    print(f"Faux positifs (FP): {fp}")
    print(f"Vrais négatifs (VN): {vn}")
    print(f"Faux négatifs (FN): {fn}")
    print(f"Taux bonne de détection : {round(100*vp/len(events))} %")

    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    added_legends = set()
    fig.add_trace(go.Scatter(
        x=df_binary.index,
        y=df_binary['binary'],
        mode='lines',
        line=dict(color='gray', width=1),
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=anomaly_times,
        y=anomalie_values,
        mode='markers',
        marker=dict(size=20, color='red', symbol='circle-open'),
        name='Anomalies',
        showlegend='Anomalies' not in added_legends
    ), row=1, col=1)


    # Points colorés par label (hors 'rotation' et 'other')
    for label in valid_labels:
        df_label = df_binary[df_binary['Label'] == label]
        show_legend = label not in added_legends
        fig.add_trace(go.Scatter(
            x=df_label.index,
            y=df_label['binary'],
            mode='markers',
            marker=dict(color=label_colors[label], size=6, opacity=0.8),
            name=label,
            showlegend=show_legend
        ), row=1, col=1)
        if show_legend:
            added_legends.add(label)

    # === Ajouter les vspan (vrects globaux pour tout le graphique) ===
    for label in df_labels['Label'].unique():
        sub_df = df_labels[df_labels['Label'] == label]
        for _, row in sub_df.iterrows():
            fig.add_vrect(
                x0=row['Start'],
                x1=row['End'],
                fillcolor='blue',
                opacity=0.5,
                line_width=0,
                layer='below',
                annotation_text=background_label,
                annotation_position="top left",
                annotation=dict(font_size=10, font_color='blue')
            )

    fig.update_layout(
        width=1500,
        height=300,
        hovermode="x unified",
        title_text=f"Impact detection after PCA - Features along time (True posifive rate : {round(100*vp/(vp+fp))} %,True negative rate : {round(100*vn/(vn+fn))} %, True detection rate : {round(100*vp/len(events))} %)"
    )

    fig.show()

    save_data = {
        "events" : events,
        "results": results,
        "filtered": filtered,
        "labels_pca": labels_pca,
        "df_labels": df_labels  
    }
    with open(f"results/plot_inputs_ch{channel}.pkl", "wb") as f:
        pickle.dump(save_data, f)



def build_labels_and_scores(pca_results, inside_points):
    labels_pca = []
    y_score_pca = []
    
    inside_points_set = set(map(tuple, inside_points))  # Conversion pour une recherche rapide
    
    for point in pca_results:
        if tuple(point) in inside_points_set:
            labels_pca.append('Etat normal')
            y_score_pca.append(0)
        else:
            labels_pca.append('Anomalie')
            y_score_pca.append(1)
    
    return np.array(labels_pca), np.array(y_score_pca)

def build_percentile_labels(results, batch, machine_name, percentile_filename=None):
    percentiles = {}
    labels_all_indicators = []

    if batch == 'train':
        # Calculate percentiles for each indicator
        for key in feature_list:
            p1 = np.nanpercentile(results[key].values, 1)
            p5 = np.nanpercentile(results[key].values, 5)
            p95 = np.nanpercentile(results[key].values, 95)
            p99 = np.nanpercentile(results[key].values, 99)
            percentiles[key] = (p1, p5, p95, p99)

    else:
        # Load percentiles from a CSV file
        if percentile_filename:
            percentiles_df = pd.read_csv(f'results/{machine_name}/{percentile_filename}', index_col=0)
            for key in results.keys():
                if key in percentiles_df.index:
                    row = percentiles_df.loc[key]
                    percentiles[key] = (row['1st_percentile'], row['5th_percentile'], row['95th_percentile'], row['99th_percentile'])

    # Define labels based on the calculated percentiles
    for key in feature_list:
        if key in percentiles:
            p1, p5, p95, p99 = percentiles[key]
            indicator_labels = np.where(
                (results[key].values < p1) | (results[key].values > p99),  # 'out' condition
                'Anomalie',
                np.where(
                    ((results[key].values >= p1) & (results[key].values < p5)) | ((results[key].values > p95) & (results[key].values <= p99)),  # 'middle' condition
                    'Etat intermédiaire',
                    'Etat normal'  # 'in' condition
                )
            )
            labels_all_indicators.append(indicator_labels)

    # Combine labels from all indicators into a single label array
    if labels_all_indicators:
        # Start with a default label
        labels = np.full(results[feature_list[0]].shape, 'Etat normal', dtype=object)  # Use the shape of one of the indicators
        for indicator_labels in labels_all_indicators:
            labels = np.where(indicator_labels == 'Anomalie', 'Anomalie', labels)
            labels = np.where(indicator_labels == 'Etat intermédiaire', 'Etat intermédiaire', labels)
    else:
        labels = np.array([])  # Return an empty array if no indicators

    return percentiles, labels

def calculate_ellipses(pca_results, labels, centroids, n_clusters,x_thresh,y_thresh):
    ellipses = []
    chi_square_99 = chi2.ppf(chi_percent/100, 2)  # Valeur critique pour 99.9% de confiance (2D)

    for i in range(n_clusters):
        # Ne cnosidérer que les points appartenant au cluster i
        cluster_points = pca_results[labels == i]

        if len(cluster_points) < 2:
            print(f"Cluster {i+1} ignoré : pas assez de points ({len(cluster_points)}).")
            continue  # Passer au cluster suivant si pas assez de points
        
        cx, cy = centroids[i]
        # Filtrer sur la position des centroïdes hors du scope de l'étude
        if x_thresh:
            if abs(cx) > x_thresh or abs(cy) > y_thresh:
                # ignore cluster whose center is out-of-bounds
                continue
    
        # Calcul de la matrice de covariance pour le cluster
        cov_matrix_i = np.cov(cluster_points, rowvar=False)

        if cov_matrix_i.shape != (2, 2):
            print(f"Cluster {i+1} ignoré : covariance invalide (shape={cov_matrix_i.shape}).")
            continue  # Passer au cluster suivant si la matrice n'est pas valide
        
        # Obtenir les valeurs propres et vecteurs propres (axes principaux de l'ellipse)
        eigenvalues_i, eigenvectors_i = np.linalg.eigh(cov_matrix_i)

        # Demi-grand axe (a) et demi-petit axe (b)
        a_i = np.sqrt(eigenvalues_i[1]) * np.sqrt(chi_square_99)  # Demi-grand axe : sqrt de la plus grande valeur propre
        b_i = np.sqrt(eigenvalues_i[0]) * np.sqrt(chi_square_99)  # Demi-petit axe : sqrt de la plus petite valeur propre

        if a_i < 0.05 or b_i < 0.05:
            continue  # Ignorez cette ellipse si l'un des axes est infinitesimal


        # Calcul de l'angle d'inclinaison en degrés
        theta_i = np.degrees(np.arctan2(eigenvectors_i[1, 1], eigenvectors_i[0, 1]))

        ellipse_obj = EllipseCluster(centroid=centroids[i], width=2*a_i, height=2*b_i, angle=theta_i)
        ellipses.append(ellipse_obj)

    return ellipses

# générer les points d'une ellipse
def generate_ellipse_points(center, width, height, angle, num_points=200):
    t = np.linspace(0, 2 * np.pi, num_points)
    a, b = width / 2, height / 2  # Demi-grand axe et demi-petit axe
    ellipse = np.array([a * np.cos(t), b * np.sin(t)])  # Points dans le repère local
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
    ])
    rotated_ellipse = rotation_matrix @ ellipse  # Rotation de l'ellipse
    return rotated_ellipse.T + center  # Translation vers le centre

# vérifier si deux ensembles de points se chevauchent
def points_overlap(points1, points2):
    # Créer des objets Path à partir des ensembles de points
    from matplotlib.path import Path
    path1 = Path(points1)
    path2 = Path(points2)
    
    # Vérifier si l'un des points de l'ensemble 1 est dans l'ensemble 2 et vice versa
    for point in points1:
        if path2.contains_point(point):
            return True
    for point in points2:
        if path1.contains_point(point):
            return True
    return False

# grouper les ellipses chevauchantes
def group_ellipses(ellipses):
    groups = []  # Liste pour stocker les groupes d'ellipses

    for ellipse in ellipses:
        added = False
        ellipse_points = generate_ellipse_points(ellipse.centroid, ellipse.width, ellipse.height, ellipse.angle)
        
        for group in groups:
            # Calculer les points pour chaque groupe et vérifier si l'ellipse chevauche ce groupe
            group_points = []
            for e in group:
                group_points.extend(generate_ellipse_points(e.centroid, e.width, e.height, e.angle))
            
            # Vérifier le chevauchement des points
            if points_overlap(ellipse_points, np.array(group_points)):
                group.append(ellipse)  
                added = True
                break

        if not added:
            groups.append([ellipse])  # Si l'ellipse ne chevauche rien, elle forme un nouveau groupe

    return groups


def classify_pca_results(pca_results, groups):
    inside_points = []
    outside_points = []
    convex_hulls = []

    # Générer les Convex Hulls pour chaque groupe
    for i, group in enumerate(groups):
        group_points = []
        for ellipse in group:
            points = generate_ellipse_points(ellipse.centroid, ellipse.width, ellipse.height, ellipse.angle)
            if len(points) > 0:
                group_points.extend(points)
        
        group_points_array = np.array(group_points)

        # Vérifie si des NaN sont présents
        if group_points_array.size == 0 or np.isnan(group_points_array).any():
            print(f"Groupe {i} ignoré : points invalides ou vides.")
            continue

        try:
            hull = ConvexHull(group_points_array)
            convex_hulls.append(group_points_array[hull.vertices])
        except Exception as e:
            print(f"Erreur lors du calcul du Convex Hull pour le groupe {i} : {e}")
            continue

    # Vérifier chaque point par rapport aux Convex Hulls
    for point in pca_results:
        is_inside = False
        for hull_points in convex_hulls:
            path = MplPath(hull_points)
            if path.contains_point(point):
                is_inside = True
                break
        if is_inside:
            inside_points.append(point)
        else:
            outside_points.append(point)

    return np.array(inside_points), np.array(outside_points), convex_hulls

def borders(pca_results, labels_percentile,ellipses,rolling_window,block_duration,threshold,batch):
    """
    Identifie les anomalies dans les résultats PCA et prépare les données pour la visualisation.
    
    pca_results : np.array -> résultats PCA à évaluer
    ellipses : list -> liste d'ellipses pour déterminer les frontières
    results : dict -> dictionnaire contenant des résultats d'analyse
    labels_percentile : array-like -> étiquettes associées aux valeurs de pourcentage
    
    Returns :
        labels_pca : list -> étiquettes 'in' ou 'out' pour chaque point PCA
        time : np.array -> vecteur de temps
        change_indices : np.array -> indices de changement
        indices_flatten : np.array -> indices aplatis
    """
    
    # Créer un vecteur de temps (de 0 à la longueur des labels)
    time = np.arange(len(labels_percentile)) * block_duration  # Vecteur de temps allant de 0 à (nombre de labels * durée par bloc)
    
    # Couleurs associées aux labels
    label_colors = {'Etat normal': 'darkolivegreen', 'Etat intermédiaire': 'chocolate', 'Anomalie': 'firebrick'}

    # Grouper les ellipses
    groups = group_ellipses(ellipses)

    # Classifier les points
    inside_points, _ ,convex_hulls = classify_pca_results(pca_results, groups)
    
    # Afficher les ellipses et les zones
    for i, group in enumerate(groups):
        group_points = []
        for ellipse in group:
            ellipse_points = generate_ellipse_points(ellipse.centroid, ellipse.width, ellipse.height, ellipse.angle)
            group_points.extend(ellipse_points)
        hull = ConvexHull(np.array(group_points))
        hull_points = np.array(group_points)[hull.vertices]
        hull_points = np.vstack([hull_points, hull_points[0]])    

    labels_pca, y_score_pca = build_labels_and_scores(pca_results, inside_points)

    y_score_percentile= []
    for label in labels_percentile:
        if label == 'Anomalie':
            y_score_percentile.append(1)
        else:
            y_score_percentile.append(0)

    # Calcul de la rolling médiane sur 1 min
    rolling_percentile = pd.Series(y_score_percentile).rolling(window=int(round(rolling_window/block_duration))).mean().fillna(0)
    rolling_pca = pd.Series(y_score_pca).rolling(window=int(round(rolling_window/block_duration))).mean().fillna(0)

    
    # Initialiser le dictionnaire alarm
    alarm = {
        'alarm_index_value_percentile': None,
        'alarm_index_value_pca': None,
        'alarm_index_sample_percentile': None,
        'alarm_index_sample_pca': None
    }
    if batch == 'test':
        if any(rolling_percentile > threshold/100):
            alarm['alarm_index_value_percentile'] = np.argmax(rolling_percentile > threshold/100)
            print(f"Premier indice où rolling_percentile > {threshold} %: échantillon n° {alarm['alarm_index_sample_percentile']}")
        if any(rolling_pca > threshold/100):
            alarm['alarm_index_value_pca'] = np.argmax(rolling_pca > threshold/100)
            print(f"Premier indice où rolling_pca > {threshold} %: échantillon n° {alarm['alarm_index_sample_pca']}")
     

    print(f"taux de valeurs anormales à partir des histogrammes : {round(100*np.mean(y_score_percentile))}%")
    print(f"taux de valeurs anormales à partir de la PCA : {round(100*np.mean(y_score_pca))}%")


    return labels_pca, time,label_colors,rolling_percentile,rolling_pca,convex_hulls,alarm

def display_histograms(data, title_suffix='', percentiles=True):
    plt.figure(figsize=(20, 5))
    pastel_colors = sns.color_palette("Set2", n_colors=len(data)) 
    features = list(data.keys())
    
    for i, key in enumerate(features):
        plt.subplot(1, len(features), i + 1)
        plt.hist(data[key], bins=2**histogram_n_bits, color=pastel_colors[i])
        plt.title(f"{key.capitalize()} histogram" + title_suffix)
        plt.xlabel(key.capitalize())
        if percentiles is not None and key in percentiles:
            plt.axvline(percentiles[key][0], color='grey', linestyle='--', label='1st percentile')
            plt.axvline(percentiles[key][1], color='k', linestyle='--', label='5th percentile')
            plt.axvline(percentiles[key][2], color='k', linestyle='--', label='95th percentile')
            plt.axvline(percentiles[key][3], color='grey', linestyle='--', label='99th percentile')
            if i == 0:
                plt.legend(loc='upper right')
            plt.grid()
    
    plt.tight_layout()
    plt.show()

def display_rolling_average(rolling_percentile,rolling_pca,time,alarm):
    

    # Création des sous-graphiques
    t0 = time[0]
    t_end = time[-1]

    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2 subplots superposés

    # Plot du rolling_percentile
    axs[0].step(time, 100*rolling_percentile)
    axs[0].set_title(f'Histogrammes')
    axs[0].set_ylabel('%')
    axs[0].grid(True)
    axs[0].set_xlim(t0, t_end)  
    axs[0].set_ylim(-1, 101)

    # Vérifier si alarm['alarm_index_percentile'] est valide
    if alarm['alarm_index_value_percentile'] is not None:
        alarm_time = time[alarm['alarm_index_value_percentile']]  # L'heure à l'indice de l'alarme
        axs[0].axvspan(alarm_time, t_end, color='red', alpha=0.3, label=f"Échantillon n° {alarm['alarm_index_sample_percentile']}")
        axs[0].legend(loc='upper right')

    # Plot du rolling_pca
    axs[1].step(time, 100*rolling_pca)
    axs[1].set_title(f'PCA')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('%')
    axs[1].grid(True)
    axs[1].set_xlim(t0, t_end)  
    axs[1].set_ylim(-1, 101)

    # Vérifier si alarm['alarm_index_pca'] est valide
    if alarm['alarm_index_value_pca'] is not None:
        alarm_time_pca = time[alarm['alarm_index_value_pca']]  # L'heure à l'indice de l'alarme PCA
        axs[1].axvspan(alarm_time_pca, t_end, color='red', alpha=0.3, label=f"Échantillon n° {alarm['alarm_index_sample_pca']}")
        axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def display_pca(pca_results,convex_hulls,data_scaled,label_colors):
    """
    pca_results : np.ndarray (n_samples, 2)
    convex_hulls: list of arrays of shape (m_i, 2)
    labels      : array-like of length n_samples with categorical labels
    label_colors: dict mapping each label -> a valid CSS color
    """
    x = pca_results[:,0]
    y = pca_results[:,1]
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    idx = density.argsort()
    x, y, density = x[idx], y[idx], density[idx]
    
    merged = unary_union([Polygon(h) for h in convex_hulls])
    polys = []
    if not merged.is_empty:
        if merged.geom_type == 'Polygon':
            polys = [merged]
        else:
            polys = list(merged.geoms)
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["PCA Density", "PCA by Label"],
                        shared_yaxes=True, shared_xaxes=False,
                        horizontal_spacing=0.1)
    
    # density scatter 
    

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            size=5,
            color=density,
            colorscale='Viridis'
        ),
        showlegend=False  # ← ici à la bonne place, à l'intérieur de go.Scatter
    ), row=1, col=1)

    for poly in polys:
        xh, yh = list(poly.exterior.xy[0]), list(poly.exterior.xy[1])
        fig.add_trace(go.Scatter(
            x=xh, y=yh,
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
            showlegend = False,
            hoverinfo='skip',
        ), row=1, col=1)
    
    # scatter par label
    label_series = data_scaled['Label'].values
    for lbl in label_colors.keys():
        mask = (label_series == lbl)
        fig.add_trace(go.Scatter(
            x=pca_results[mask, 0],
            y=pca_results[mask, 1],
            mode='markers',
            marker=dict(size=6, color=label_colors[lbl]),
            name=lbl
        ), row=1, col=2)
    
    for poly in polys:
        xh, yh = list(poly.exterior.xy[0]), list(poly.exterior.xy[1])
        fig.add_trace(go.Scatter(
            x=xh, y=yh,
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
            hoverinfo='skip',
            showlegend=False  # déjà affiché à gauche
        ), row=1, col=2)
    
    fig.update_layout(
        title="PCA: Density vs Labels",
        width=1200, height=500,
        hovermode="closest"
    )
    fig.update_xaxes(title="PC1", row=1, col=1)
    fig.update_xaxes(title="PC1", row=1, col=2)
    fig.update_yaxes(title="PC2", row=1, col=1)
    fig.update_yaxes(title="PC2", row=1, col=2)
    
    fig.show()

def merge_vspans(df_labels, max_gap=pd.Timedelta('5s')):
    merged = []
    for label, group in df_labels.groupby("Label"):
        group = group.sort_values("Start")
        current_start = group.iloc[0]['Start']
        current_end = group.iloc[0]['End']

        for i in range(1, len(group)):
            row = group.iloc[i]
            if row['Start'] - current_end <= max_gap:
                current_end = max(current_end, row['End'])
            else:
                merged.append((label, current_start, current_end))
                current_start = row['Start']
                current_end = row['End']
        merged.append((label, current_start, current_end))
    return pd.DataFrame(merged, columns=['Label', 'Start', 'End'])


#################
# COMPILATION ###
#################
os.chdir(path)
print(path)

# === Charger les labels ===
label_dir = Path(f"data/{sensor}/{date}/long_format/labels")
label_files = list(label_dir.rglob("labels.csv"))

df_list = [pd.read_csv(f) for f in label_files]
df_labels = pd.concat(df_list, ignore_index=True)
df_labels['Start'] = pd.to_datetime(df_labels['Start'])
df_labels['End'] = pd.to_datetime(df_labels['End'])
df_labels = df_labels[df_labels['Label'] == background_label]
df_labels = df_labels.drop_duplicates(subset='Start').sort_values(by='Start')
df_labels = merge_vspans(df_labels,max_gap=pd.Timedelta(seconds = 30))

# === Charger les données stéréo ===
if batch == 'train':
    df1 = pd.read_csv(f"results/{sensor}/{sensor_id}/ch{channel}/dataframe_features.csv", parse_dates=['time'])
    df2 = pd.read_csv(f"results/{sensor}/{sensor_id}/ch{channel+2}/dataframe_features.csv", parse_dates=['time'])
    df = pd.concat([df1, df2], ignore_index=True)
else:
    df = pd.read_csv(f"results/{sensor}/{sensor_id}/ch{channel}/dataframe_features.csv", parse_dates=['time'])
df.set_index('time', inplace=True)
if feature_list == 'all':
    feature_list = df.columns[df.columns.get_loc('Label') + 1:].tolist()
df = df[['Label'] + feature_list]
df = df.dropna(subset=feature_list)

df['Label'] = df['Label'].replace('impact_inconnu', 'inconnu')
df['Label'] = df['Label'].apply(
    lambda x: "rivets (pas d'impact)" if 'no-impact' in x.lower() else x
)
df['Label'] = df['Label'].replace('rivets_ch1-2', 'rivets')
df['Label'] = df['Label'].replace('rivets_ch-1-2', 'rivets')
df['Label'] = df['Label'].replace('rivets_ch-3-4', 'rivets')
df['Label'] = df['Label'].replace('rivets_milieu', 'rivets')
df['Label'] = df['Label'].replace(' pneumatique', 'pneumatique')
# Retouche pour éviter les effets de bords des recordings
df.loc[df['ultrasoundlevel'] < 30, 'ultrasoundlevel'] = 30

# =======================
# === Préparation Data ===
# =======================
# Sauvegarder l’ordre initial
initial_order = df.index.copy()

# Trier par datetime croissant
df_sorted = df.sort_index()

# ===============================
# === Nettoyage autour des NaN ===
# ===============================
cols_suppl = df_sorted.columns.difference(['Label'])
df_temp = df_sorted.reset_index()
time_diffs = df_temp['time'].diff().dt.total_seconds()
jump_positions = time_diffs[time_diffs > 1].index
mask_zero = pd.Series(False, index=df_temp.index)
for pos in jump_positions:
    # ligne avant le saut (pos - 1) si possible
    if pos > 0:
        mask_zero.iloc[pos - 1] = True
    # ligne du saut (pos)
    mask_zero.iloc[pos] = True
    # ligne après le saut (pos + 1) si possible
    if pos < len(df_temp) - 1:
        mask_zero.iloc[pos + 1] = True
df_temp.loc[mask_zero, cols_suppl] = 0
df_sorted = df_temp.set_index('time')

# ============================================================================
# === Extension des labels d’impacts lisser les imprécisions d'annotation  ===
# ============================================================================
df_src = df_sorted.reset_index()
df_out = df_src.copy()

# Identifier les lignes avec labels "impact" (≠ 'rotation', 'other')
target_mask = ~df_src['Label'].isin(['rotation', 'other'])
positions = df_src[target_mask].index

# Étendre les annotations avant les impacts
for pos in positions:
    label = df_src.loc[pos, 'Label']
    for offset in [-2,-1]:
        new_pos = pos + offset
        if 0 <= new_pos < len(df_out):
            current = df_src.loc[new_pos, 'Label']
            if current in ['rotation', 'other']:
                df_out.loc[new_pos, 'Label'] = label

df_modified = df_out.set_index('time')
df_sorted = df_modified.loc[initial_order]

# =========================================
# === Attribution des couleurs par label ==
# =========================================

excluded_labels = ['rotation', 'other']
valid_labels = [lbl for lbl in df['Label'].unique() if lbl not in excluded_labels]

# Palette commune
color_palette = px.colors.qualitative.Set2
label_colors = {label: color_palette[i % len(color_palette)] for i, label in enumerate(valid_labels)}

# Couleur spécifique pour "other"
label_colors['other'] = 'rgba(0,0,0,1)'  # noir opaque

# ==============================
# === Détection des impacts ===
# ==============================
# Tri + suppression des doublons
df_sorted = df_sorted.sort_index().copy()
df_sorted = df_sorted[~df_sorted.index.duplicated(keep='first')]

results = df_sorted[df_sorted['ultrasoundlevel']>0].copy()

if features_to_display == 'all':
    features_to_display = feature_list

scaler = StandardScaler()


time_array = results.index  # ou df['timestamp'] si c'est une colonne

block_duration = (time_array[1] - time_array[0]).total_seconds()

if batch == 'train':

    # filtrer les lignes sans 'rivets' ni 'cutter'
    mask = ~results['Label'].str.contains("rivets|cutter|rivets (pas d'impact)", case=False, na=False)
    filtered = results[mask].copy()
    numeric_cols = filtered.select_dtypes(include='number').columns
    data_scaled_array = scaler.fit_transform(filtered[numeric_cols])

    # Sauvegarder les coefficients de normalisation
    coefficients_df = pd.DataFrame({
        'mean': scaler.mean_,
        'scale': scaler.scale_
    })
    os.makedirs(f'results/{machine_name}', exist_ok=True)
    coefficients_df.to_csv(f'results/{machine_name}/normalization_coefficients.csv', index=False)

    data_scaled = pd.DataFrame(data_scaled_array, columns=numeric_cols, index=filtered.index)

    valid_data = data_scaled[~np.isnan(data_scaled).any(axis=1)]  # Exclure les lignes contenant des NaN
    data_scaled['Label'] = filtered['Label']

    pca = PCA(n_components=2)  # Choisir le nombre de composantes principales
    pca_results = pca.fit_transform(valid_data)  

    loadings = pca.components_.T  
    impact_pc1 = np.abs(loadings[:,0])  # Impact pour la 1ère composante principale
    impact_pc2 = np.abs(loadings[:,1])  # Impact pour la 2ème composante principale

    # Somme des impacts pour chaque paramètre (en termes d'importance totale dans la projection)
    contributions = impact_pc1 + impact_pc2

    contributions = pd.DataFrame({
        'Feature': feature_list,
        'Contributions': contributions
    })
    contributions = contributions.sort_values(by='Contributions', ascending=False)


    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(pca_results)
    centroids = kmeans.cluster_centers_

    ellipses = calculate_ellipses(pca_results, labels, centroids, n_clusters,x_thresh,y_thresh)

    with open(f'results/{machine_name}/ellipses.pkl', 'wb') as file:
        pickle.dump(ellipses, file)

    # Sauvegarder la matrice de transformation PCA
    np.save(f'results/{machine_name}/pca_components.npy', pca.components_)  # Sauvegarder les vecteurs propres PCA


else:

    mask = results['Label'].str.contains("rivets|cutter|rivets (pas d'impact)", case=False, na=False)
    filtered = results[mask].copy()
    numeric_cols = filtered.select_dtypes(include='number').columns

    # Charger les coefficients de normalisation
    coefficients_df = pd.read_csv(f'results/{machine_name}/normalization_coefficients.csv')
    scaler.mean_ = coefficients_df['mean'].values
    scaler.scale_ = coefficients_df['scale'].values

    # Appliquer le scaler sur les données de test
    if filtered.shape[0] == 0:
        raise ValueError("No data to scale. Verify input to StandardScaler.")
    data_scaled_array = scaler.transform(filtered[numeric_cols])

    # Charger les vecteurs propres PCA depuis le fichier
    pca_components = np.load(f'results/{machine_name}/pca_components.npy')  # Charger les vecteurs propres PCA
    contributions = None
    data_scaled = pd.DataFrame(data_scaled_array, columns=numeric_cols, index=filtered.index)

    valid_data = data_scaled[~np.isnan(data_scaled).any(axis=1)]  # Exclure les lignes contenant des NaN


    # Étape 5 : ajouter la colonne 'Label' non modifiée
    data_scaled['Label'] = filtered['Label']
    pca_results = np.dot(valid_data, pca_components.T)
    pca_with_nan = np.full((data_scaled.shape[0], pca_results.shape[1]), np.nan)
    valid_mask = ~np.isnan(data_scaled_array).any(axis=1)  # Masque pour lignes valides (sans NaN)
    pca_with_nan[valid_mask] = pca_results  # Remplir avec les résultats PCA

    with open(f'results/{machine_name}/ellipses.pkl', 'rb') as file:
        ellipses = pickle.load(file)


raw_percentiles, _ = build_percentile_labels(filtered, batch=batch,machine_name=machine_name,percentile_filename='raw_percentiles.csv')
standardized_percentiles, labels_percentile = build_percentile_labels(data_scaled, batch=batch,machine_name=machine_name,percentile_filename='standardized_percentiles.csv')


if batch == 'train':
    save_percentiles_to_csv(raw_percentiles,machine_name, 'raw_percentiles.csv')    
    save_percentiles_to_csv(standardized_percentiles,machine_name, 'standardized_percentiles.csv')

if features_to_display:
    # Afficher les histogrammes des données brutes
    results_hist = {key: value for key, value in filtered.items() if key in features_to_display}
    results_flatten = {key: value.values.flatten() for key, value in results_hist.items()}
    display_histograms(results_flatten, title_suffix=' (Raw Data)', percentiles=raw_percentiles)

    # Afficher les histogrammes des données standardisées
    filtered_data_scaled_dict = {key: value for key, value in data_scaled.items() if key in features_to_display}
    data_scaled_dict_flatten = {key: value.values.flatten() for key, value in filtered_data_scaled_dict.items()}
    display_histograms(data_scaled_dict_flatten, title_suffix=' (Standardized Data)', percentiles=standardized_percentiles)

labels_pca, time,_,rolling_percentile,rolling_pca,convex_hulls,alarm = borders(pca_results, labels_percentile,ellipses,rolling_window,block_duration,threshold,batch)  

excluded_labels = ['rotation', 'other']
valid_labels = [lbl for lbl in results['Label'].unique() if lbl not in excluded_labels]

# Palette commune
color_palette = px.colors.qualitative.Set2
label_colors = {label: color_palette[i % len(color_palette)] for i, label in enumerate(valid_labels)}


display_rolling_average(rolling_percentile,rolling_pca,time,alarm)

display_pca(pca_results,convex_hulls,data_scaled,label_colors)

if batch == 'test':
    display_anomaly_detector(results,filtered,labels_pca,label_colors)


# %%
