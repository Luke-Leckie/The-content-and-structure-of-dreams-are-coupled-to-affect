#Author Luke Leckie. Code for publication entitled "The content and structure of dreams are coupled to affect "
import networkx as nx
import signal
from networkx.algorithms import community
import community
from community import community_louvain
import random
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from gensim.models import Word2Vec 
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, make_scorer
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

stats = importr('stats')










def structureG (cell, dictionary):#wwhere cell is an atom sequence
    G=nx.DiGraph()
    edge_list=[]
    numbers = cell.split(' ')[0:]  
    cell = [int(num) for num in numbers]
    for i in range(0, len(cell)-1):
        topic1=cell[i]
        topic2=cell[i+1]
        if topic1==' 'or topic2==' ' or np.isnan(topic1) or np.isnan(topic2):
            continue
        if topic1==topic2:
            continue
        else:
            edge_list.append(tuple([topic1, topic2]))
    edge_weights = defaultdict(lambda: {'count': 0, 'inverse_weight': 0, 'direct_weight': 0})
    for edge in edge_list:
        # We use count weights for basic mod.
        edge_weights[edge]['count'] += 1
        cos_sim = cosine_similarity(dictionary[edge[0]].reshape(1, -1), dictionary[edge[1]].reshape(1, -1))[0][0]
        #We make inverse and direct weights, which are used for different network measures
        inverse_weight = 1 / (cos_sim + 1)
        direct_weight = cos_sim + 1
        edge_weights[edge]['inverse_weight'] = inverse_weight
        edge_weights[edge]['direct_weight'] = direct_weight
    # Add edges with all weights to the graph
    for edge, weights in edge_weights.items():
        G.add_edge(edge[0], edge[1], count=weights['count'], inverse_weight=weights['inverse_weight'], direct_weight=weights['direct_weight'])

    return G
def structureG_simple (cell, dictionary):#wwhere cell is an atom sequence
    '''Same function but without cosine so less computationally heavy'''
    G=nx.DiGraph()
    edge_list=[]
    numbers = cell.split(' ')[0:]  
    cell = [int(num) for num in numbers]
    for i in range(0, len(cell)-1):
        topic1=cell[i]
        topic2=cell[i+1]
        if topic1==' 'or topic2==' ' or np.isnan(topic1) or np.isnan(topic2):
            continue
        if topic1==topic2:
            continue
        else:
            edge_list.append(tuple([topic1, topic2]))
    edge_weights = defaultdict(lambda: {'count': 0, 'inverse_weight': 0, 'direct_weight': 0})
    for edge in edge_list:
        edge_weights[edge]['count'] += 1
    for edge, weights in edge_weights.items():
        G.add_edge(edge[0], edge[1], count=weights['count'])

    return G
        
def get_first(cell):
    numbers = cell.split(' ')[0:]
    cell = [int(num) for num in numbers]
    return cell[0]
def get_last(cell):
    numbers = cell.split(' ')[0:] 
    cell = [int(num) for num in numbers]
    return cell[-1]
    
    
def timeout_handler(signum, frame):
    raise TimeoutError("Modularity calculation took too long")

def degree_heterogeneity(G, weighted=False):
    '''Computes the coefficient of variation of degree in a graph'''
    if weighted:
        degrees = [d for n, d in G.degree(weight='weight')]
    else:
        degrees = [d for n, d in G.degree()]
    # Compute the mean and standard deviation of the degrees
    mean_degree = sum(degrees) / len(degrees)
    std_dev_degree = (sum([(d - mean_degree) ** 2 for d in degrees]) / len(degrees)) ** 0.5
    # Compute the coefficient of variation
    cv = std_dev_degree / mean_degree
    return cv
def feedback(G):
    '''Computes the proportion of edges with feedback loops'''
    feedbacks = 0
    checked_pairs = set()
    for edge in G.edges():
        if (edge[1], edge[0]) in G.edges() and edge not in checked_pairs:
            feedbacks += 1
            checked_pairs.add(edge)
            checked_pairs.add((edge[1], edge[0]))
    total_edges = G.number_of_edges()
    proportion_feedbacks = feedbacks / total_edges
    return proportion_feedbacks

def handler(signum, frame):
    raise Exception("Timeout!")
def run_louvain_with_timeout(G, weighting='weight', timeout=10):
    '''Custom method to run Louvain algorithm. I use this so that if louvain does not compute in 10
    for a graph it reruns with a new random seed. This is important as on certain random seeds and graphs
    Louvain can get stuck indefinitely'''
    signal.signal(signal.SIGALRM, handler)
    partition_found = False
    attempts = 0
    while not partition_found:
        try:            
            seed = random.randint(0, 10000)
            signal.alarm(timeout)  # Set the timeout
            partitions = nx.community.louvain_communities(G,weight=weighting, seed=seed)
            signal.alarm(0)  
            partition_found = True
            return partitions
        except TimeoutError as e:
            attempts += 1
            signal.alarm(0)  
            if attempts >= 5: 
                return None
def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def get_net_measures(G):
    '''Compute your network measures'''
    nodes=len(G.nodes())
    edges=len(G.edges())
    try:
        fb=feedback(G)
        clustering=nx.average_clustering(G)
        transitivity=nx.transitivity(G)
        deg_het=degree_heterogeneity(G)
        Gi=gini_coefficient(np.array([degree for node, degree in G.degree()])+0.1)
        G2=G.to_undirected(reciprocal=False)
        efficiency=nx.global_efficiency(G2)
        comms = run_louvain_with_timeout(G, weighting='direct_weight')
        community_num=len(comms)
        cosine_mod=nx.community.modularity(G,comms, weight='direct_weight')
        comms = run_louvain_with_timeout(G, weighting='count')
        mod=nx.community.modularity(G,comms, weight='count')
        diameter=nx.diameter(G2, weight=None)
        av_path=nx.average_shortest_path_length(G2, weight=None)
    except Exception as e:
        clustering=np.nan
        deg_het=np.nan
        efficiency=np.nan
        density=np.nan
        cosine_mod=np.nan
        transitivity=np.nan
        community_num=np.nan
        mod=np.nan
        fb=np.nan
        Gi=np.nan
        diameter=np.nan
        av_path=np.nan
    return nodes, edges, clustering, transitivity,diameter, av_path, deg_het,\
efficiency, Gi, community_num, mod,cosine_mod,fb

def calculate_measures(G):
    nn, en, cl, tr, di, ap, dh, ef, Gi,co, mo,como, fb = get_net_measures(G)
    return pd.Series({
        'Nodes': nn,
        'Edges': en,
        'Clustering': cl,
        'Transitivity': tr,
        'Diameter': di,
        'AP': ap,
        'Degree Het.': dh,
        'Efficiency': ef,
        'Gini': Gi,
        'Communities': co,
        'Modularity': mo,
        'Cosine mod.':como,
        'Feedback loops': fb
    })
    
def calculate_paths(row):
    '''Calculates start to end path with and without inverse cosine weighting'''
    G = row['graphs']
    if len(G.nodes()) == 0:
        return pd.Series({'Topic path': np.nan, 'Cosine path': np.nan})
    first = row['first']
    last = row['last']
    path = len(nx.shortest_path(G, source=first, target=last))
    path_cosine = len(nx.shortest_path(G, source=first, target=last, weight='inverse_weight'))
    return pd.Series({'Topic path': path, 'Cosine path': path_cosine})    

def creat_vec(word_list, w2vmodel):
    vec=np.mean([w2vmodel.wv[word] for word in word_list], axis=0)
    return vec
    
def calculate_median_affect(narrative, key):
    topics = narrative.split()
    valences = [key.get(topic) for topic in topics]
    if valences:
        return np.median(valences)  # Return a float
    else:
        return np.nan  # Default valence for narratives with no topics
        
def print_topics(list_keys, topn=5, reverse=False):
    if not reverse:
        for i in range(0, len(list_keys)):
            if list_keys[i-1]==list_keys[i]:
                continue
            print(list_keys[i])
            print([i[0] for i in w2vmodel.wv.similar_by_vector(dictionary[int(list_keys[i])], topn=topn)])
def get_weights (df):
    '''Extract coherence of connections'''
    all_weights=[]
    for G in df['graphs']:
        direct_weights = [data['direct_weight']-1 for _, _, data in G.edges(data=True)]
        all_weights.append(direct_weights)
    df['weights']=all_weights
    return df
    
    
def get_optimal_components_via_cv(df, columns_to_plot, param, n_components_range=range(1, 10), n_splits=5):
    df_cleaned = df.dropna(subset=columns_to_plot + [param])
    df_cleaned = df_cleaned[np.isfinite(df_cleaned[param])]

    X = df_cleaned[columns_to_plot].values
    y = df_cleaned[param].values.reshape(-1, 1)
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = y.ravel()
    mse_scores = []
    r2_scores = []
    r2_scores_2=[]
    for n_components in n_components_range:
        pls = PLSRegression(n_components=n_components)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2_cv_scores = cross_val_score(pls, X, y, cv=kf, scoring='r2')
        r2_s = r2_cv_scores.mean()
        print('Components:',n_components, 'R2:',r2_s)

def get_optimal_components_via_loo(df, columns_to_plot, param, n_components_range=range(1, 10), n_splits=5):
    '''function to determine optimum number of PLS components'''
    df_cleaned = df.dropna(subset=columns_to_plot + [param])
    df_cleaned = df_cleaned[np.isfinite(df_cleaned[param])]
    X = df_cleaned[columns_to_plot].values
    y = df_cleaned[param].values.reshape(-1, 1)
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = y.ravel()
    mse_scores = []
    r2_scores = []
    r2_scores_2=[]
    for n_components in n_components_range:
        pls = PLSRegression(n_components=n_components)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        loo = LeaveOneOut()
        mse_scorer = make_scorer(mean_squared_error)
        mse_cv_scores = cross_val_score(pls, X, y, cv=loo, scoring=mse_scorer)
        mean_mse = mse_cv_scores.mean()
        rmse = np.sqrt(mean_mse)
        baseline_pred = np.mean(y)
        baseline_mse = mean_squared_error(y, [baseline_pred] * len(y))
        baseline_rmse = np.sqrt(baseline_mse)
        # Print the results
        print('Components:', n_components)
        print('RMSE:', rmse)
        print('Baseline RMSE:', baseline_rmse)
        print('Improvement:',(baseline_rmse-rmse)/baseline_rmse)
        # Compare model RMSE with baseline RMSE
        if rmse < baseline_rmse:
            print('The model performs better than the baseline.')
        else:
            print('The model does not perform better than the baseline.')
            
def get_pls_coefficients(df, columns_to_plot, param, optimal_components=1, n_permutations=1000):
    '''function to get PLS coeffs with permuted p values'''
    df_cleaned = df.dropna(subset=columns_to_plot + [param])
    df_cleaned = df_cleaned[np.isfinite(df_cleaned[param])]
    X = df_cleaned[columns_to_plot].values
    y = df_cleaned[param].values.reshape(-1, 1)
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = y.ravel()
    pls = PLSRegression(n_components=optimal_components)
    pls.fit(X, y)
    coefficients = pls.coef_.ravel()
    # Initialize a list to store p-values for each coefficient
    perm_coefficients = np.zeros((n_permutations, len(columns_to_plot)))
    for i in range(n_permutations):
        # Permute y and fit the model
        y_permuted = np.random.permutation(y)
        pls_permuted = PLSRegression(n_components=optimal_components)
        pls_permuted.fit(X, y_permuted)
        perm_coefficients[i, :] = pls_permuted.coef_.ravel()
    # Calculate p-values for the coefficients
    p_values = []
    for i in range(len(columns_to_plot)):
        p_value = np.mean(np.abs(perm_coefficients[:, i]) >= np.abs(coefficients[i]))
        p_values.append(p_value)
    p_adjust = stats.p_adjust(FloatVector(p_values), method = 'BH')
    # Return a DataFrame with the results
    return pd.DataFrame({
        'Variable': columns_to_plot,
        'Coefficients': coefficients,
        'P-Values': p_adjust
    })

def get_percentile_indices(column, percentile, top=True):
    values = corpus[column].dropna().values
    threshold = np.percentile(values, 100 - percentile if top else percentile)
    if top:
        indices = np.where(corpus[column] >= threshold)[0]
    else:
        indices = np.where(corpus[column] <= threshold)[0]
    return indices
