import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import MAPE_loss

def boxplot_node_metric(df, node_idx, network_simul, metric_name):
    plt.figure(figsize=(3,7))
    sns.boxplot(df.iloc[node_idx][:-1], showmeans=True)
    plt.xlabel(f"{network_simul.network_graph.nodes[node_idx]['title']} (ligne {network_simul.network_graph.nodes[node_idx]['group']})")
    plt.ylabel(f'{metric_name}')

def boxplot_node_metric_per_line(df, line, network_simul, metric_name):
    plt.figure(figsize=(10,5))
    df_boxplot_line = df[df['line'] == str(line)].drop(columns='line')
    df_boxplot_line = df_boxplot_line.T
    boxplot = sns.boxplot(df_boxplot_line, orient='h', showmeans=True)
    boxplot.set_yticklabels((network_simul.network_graph.nodes[node_idx]['title'] for node_idx in df_boxplot_line.columns))
    plt.xlabel(f'{metric_name}')

def plot_true_predicted(predicted_flows, actual_flows, removed_edges_name=None):
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=actual_flows, y=predicted_flows)
    plt.plot([min(actual_flows), max(actual_flows)],
            [min(actual_flows), max(actual_flows)],
            color='red', linestyle='--')
    
    removed_title = f" with {', '.join(removed_edges_name)} removed" if removed_edges_name is not None else ''
    plt.xlabel("Actual Passenger Flow")
    plt.ylabel("Predicted Passenger Flow")
    plt.title(f"Actual vs Predicted Passenger Flow{removed_title}", wrap=True)

def plot_predicted_ape(predicted_flows, actual_flows, removed_edges_name=None):
    mape = MAPE_loss(reduction='none')
    ape_predictions = mape(predicted_flows, actual_flows)
    
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=actual_flows, y=ape_predictions)
    plt.plot([min(actual_flows), max(actual_flows)],
            [0, 0],
            color='red', linestyle='--')  # Perfect match line
    plt.xlabel("Actual Passenger Flow")
    plt.ylabel("Absolute Percentage Error")
    removed_title = f" with {', '.join(removed_edges_name)} removed" if removed_edges_name is not None else ''
    plt.title(f"APE with respect to the reference passenger flow{removed_title}")