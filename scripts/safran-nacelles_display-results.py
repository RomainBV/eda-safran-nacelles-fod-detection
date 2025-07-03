#%%
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from collections import Counter,defaultdict


channel_ref = 1

# --- Charger une première fois pour obtenir les features ---
with open(f"results/plot_inputs_ch{channel_ref}.pkl", "rb") as f:
    data_init = pickle.load(f)

results_init = data_init["results"]
feature_columns = [col for col in results_init.columns if col != 'Label']
n_features = len(feature_columns)

# === Créer la figure avec n_features + 2 lignes (2 lignes de "Detected Events") ===
fig = make_subplots(
    rows=n_features + 2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    subplot_titles=feature_columns + ['Detected Events - Ch1', 'Detected Events - Ch2']
)

all_events = []  # stockera tous les événements
# === Boucle sur les deux canaux ===
for idx in range(2):


    with open(f"results/plot_inputs_ch{(idx+1)*channel_ref}.pkl", "rb") as f:
        data = pickle.load(f)

    results = data["results"]

    cols_suppl = results.columns.difference(['Label'])
    df_temp = results.reset_index()
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
    results = df_temp.set_index('time')

    events = data['events']
    filtered = data["filtered"]
    labels_pca = data["labels_pca"]
    df_labels = data["df_labels"]


    for ev in events:
        all_events.append({
            'channel': idx + 1,         # canal 1 ou 2
            'start': ev['start'],       # index (entier)
            'end': ev['end'],           # index (entier)
            'labels': ev['labels'],
            'index': results.index      # index temporel (pandas)
        })

    excluded_labels = ['rotation', 'other']
    valid_labels = [lbl for lbl in results['Label'].unique() if lbl not in excluded_labels]

    color_palette = px.colors.qualitative.Set2
    label_colors = {label: color_palette[i % len(color_palette)] for i, label in enumerate(valid_labels)}
    label_colors['other'] = 'rgba(0,0,0,1)'

    df_binary = pd.DataFrame(results['Label']).copy()
    df_binary['binary'] = results['Label'].isin(['cutter', 'rivets', "rivets (pas d'impact)"]).astype(int)

    anomaly_times = filtered.index[labels_pca == 'Anomalie']
    row_offset = n_features + idx + 1
    added_legends = set() if idx == 0 else set(valid_labels + ['Anomalies'])

    # === Tracer les features une seule fois ===
    for i, feature in enumerate(feature_columns):
        color = 'purple' if idx == 0 else 'darkorange'
        label = 'Ch1' if idx == 0 else 'Ch2'

        fig.add_trace(go.Scatter(
            x=results.index,
            y=results[feature],
            mode='lines',
            line=dict(color=color, width=1),
            showlegend=False
        ), row=i+1, col=1)

        if feature == 'spectralflux':
            # Échelle logarithmique pour spectral_flux
            fig.update_yaxes(
                type='log',
                title_text='no dim.',
                row=i+1, col=1
            )
        elif feature == 'ultrasoundlevel':
            # échelle linéaire, unité dB
            fig.update_yaxes(
                type='linear',
                title_text='dB',
                row=i+1, col=1
            )
        else:
            # échelle linéaire, pas d’unité
            fig.update_yaxes(
                type='linear',
                title_text='no dim.',
                row=i+1, col=1
            )

        # Points colorés par label
        for label in valid_labels:
            df_label = results[results['Label'] == label]
            show_legend = label not in added_legends
            fig.add_trace(go.Scatter(
                x=df_label.index,
                y=df_label[feature],
                mode='markers',
                marker=dict(color=label_colors[label], size=6, opacity=0.8),
                name=label,
                showlegend=show_legend
            ), row=i+1, col=1)
            if show_legend:
                added_legends.add(label)


    # === Légende “Features – Ch1” et “Features – Ch2” (ajout d'un tracé imaginaire pour l'affichage correct des légendes) ===
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='purple', width=2),
        name='Features - Ch1',
        showlegend = (idx == 0)
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='darkorange', width=2),
        name='Features - Ch2',
        showlegend = (idx == 0)
    ))
    # Ligne binaire
    fig.add_trace(go.Scatter(
        x=df_binary.index,
        y=df_binary['binary'],
        mode='lines',
        line=dict(color='gray', width=1),
        showlegend=False
    ), row=row_offset, col=1)

    # Anomalies
    fig.add_trace(go.Scatter(
        x=anomaly_times,
        y=[1] * len(anomaly_times),
        mode='markers',
        marker=dict(size=20, color='red', symbol='circle-open'),
        name='Anomalies',
        showlegend='Anomalies' not in added_legends
    ), row=row_offset, col=1)

    # Points colorés par label
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
        ), row=row_offset, col=1)
        if show_legend:
            added_legends.add(label)

    # Vrects
    for label in df_labels['Label'].unique():
        sub_df = df_labels[df_labels['Label'] == label]
        for _, row in sub_df.iterrows():
            fig.add_vrect(
                x0=row['Start'],
                x1=row['End'],
                # Bleu clair en RGBA avec alpha 0.1 (très transparent)
                fillcolor='rgba(173, 216, 230, 0.1)',
                # ou si tu préfères définir une opacité séparée :
                opacity=0.1,
                line_width=0,
                layer='below'
            )


# Compteurs par label
total_by_label = defaultdict(int)
anomalies_by_label = defaultdict(int)

# Pour éviter les doublons : stocke les (label, time) déjà comptés
seen_anomalies = set()

for ev in all_events:
    ch = ev['channel']
    label = ev['labels'][0]
    start, end = ev['start'], ev['end']
    idx = ev['index']
    time_range = idx[start:end]

    # Index anomalies canal
    if ch == 1:
        anomaly_idx = data_init["filtered"].index[data_init["labels_pca"] == 'Anomalie']
    else:
        with open(f"results/plot_inputs_ch{2*channel_ref}.pkl", "rb") as f:
            data_ch2 = pickle.load(f)
        anomaly_idx = data_ch2["filtered"].index[data_ch2["labels_pca"] == 'Anomalie']

    # Incrémenter le nombre total d'événements de ce type
    total_by_label[label] += 1

    # Vérifie si une nouvelle anomalie (non comptée pour ce label)
    for t in time_range:
        if t in anomaly_idx and (label, t) not in seen_anomalies:
            anomalies_by_label[label] += 1
            seen_anomalies.add((label, t))
            break  # une seule anomalie suffit à valider l'événement

print("\nAnomalies par type d'événement :")
for label in total_by_label:
    total = 0.5*total_by_label[label]
    anomalies = anomalies_by_label[label]
    ratio = anomalies / total if total > 0 else 0
    print(f"{label}: {anomalies} anomalies sur {total} événements ({ratio:.1%})")


# === Layout final ===
fig.update_layout(
    width=1500,
    height=(n_features + 2) * 200,
    hovermode="x unified",
    title_text="Anomaly detection along time using PCA analysis"
)

fig.show()
# %%
