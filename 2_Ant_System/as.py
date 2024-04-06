import copy    
import numpy as np
import random

def get_graph():
    nodes = ['A', 'B', 'C', 'D']
    edges = [[-1, 3, 1, 2], [3 ,0, 5,4],[1, 5, 2,2],[2,4,2,-1]]

    edges = np.array(edges, dtype=np.float64)
    for i in range(len(edges)):
        edges[i][i] = np.inf
    return nodes, edges



def get_random_hamiton_path(nodes, edges):
    path = []
    start_node = random.choice(range(len(nodes)))
    path.append(start_node)
    path_w = []
    
    for _ in range(len(edges)-1):
        
        last_node = path[-1]

        min_index = -1
        min_value = float('inf')

        for i, w in enumerate(edges[last_node]):
            if i == last_node: continue
            if i in path: continue

            if w < min_value:
                min_value = w
                min_index = i
        
        path.append(min_index)
        path_w.append(min_value)
    path.append(path[0])
    path_w.append(edges[path[-2]][path[-1]])
    return sum(path_w)
            
    
def get_init_pheromone(nodes, edges):
    greed_short_len = get_random_hamiton_path(nodes, edges)
    print("greed short len", greed_short_len)
    phers = np.ones_like(edges) / greed_short_len
    np.fill_diagonal(phers, 0)
    return phers

def get_sum_prob(scores):
    scores = np.array(scores)
    scores /= np.sum(scores)
    # print("score", scores)
    sum_probs = np.zeros_like(scores)

    sum_probs[0] = scores[0]
    for i in range(1, len(scores)):
        sum_probs[i] = sum_probs[i-1] + scores[i]
    return sum_probs

def get_next_node(invisit_nodes):
    a = 1
    b = 2
    # print("Get next node")
    # print(invisit_nodes)

    invisit_nodes = np.array(invisit_nodes)
    node_index = invisit_nodes[:, 0]
    edges = invisit_nodes[:, 1]
    phers = invisit_nodes[:, 2]

    
    socre = ( (1 / edges) ** b ) * (phers ** a) 
    score = get_sum_prob(socre)

    prob = random.uniform(0, 1)
    
    select = np.where(score>prob)[0][0]
    # print(score, prob, select)
    return invisit_nodes[select]

def get_hamiton_path(nodes, edges, phers, path:list):
    for _ in range(len(edges)-1):
        last_node = path[-1]
        invisit_nodes = []
        for i, w in enumerate(edges[last_node]):
            if i == last_node: continue
            if i in path: continue

            invisit_nodes.append((i, edges[last_node][i], phers[last_node][i]))
            
        if len(invisit_nodes) == 1:
            path.append(invisit_nodes[0][0])
        else:
            next_node = get_next_node(invisit_nodes)
            path.append(int(next_node[0]))
    path.append(path[0])
    return path

def ant_sys():
    n_ants = 3

    p = 0.9

    nodes, edges = get_graph()
    phers = get_init_pheromone(nodes, edges)

    epochs = 10


    min_path = None
    min_path_w = float('inf')

    for _ in range(epochs):
        print(f"==================== epoch {_} =======================")
        start_nodes = random.sample(range(len(nodes)), n_ants)
        ants_path = [[v] for i, v in enumerate(start_nodes)]
        ants_path_w = np.zeros([n_ants])
        # 1. 蚂蚁群体出动，按照信息素和启发式信息寻找路径
        for i, ant_path in enumerate(ants_path):
            # print(f"ant {i} start", ant_path)
            ant_path = get_hamiton_path(nodes, edges, phers, ant_path)
            ants_path_w[i] = sum([edges[ant_path[j]][ant_path[j+1]] for j in range(len(ant_path)-1)])
            print(f"ant {i} end", ant_path, "path len", ants_path_w[i])
        
        cur_min_path_w = np.min(ants_path_w)
        if cur_min_path_w < min_path_w:
            min_path = ants_path[np.argmin(ants_path_w)]
            min_path_w = cur_min_path_w
        
        print(f"shortest path : {min_path}, {min_path_w}")

        #2. 更新信息素
        ants_path_score = 1 / ants_path_w
        delta_phers = np.zeros_like(phers)
        for i in range(n_ants):
            ant_path = ants_path[i]
            ant_path_score = ants_path_score[i]

            for i in range(len(ant_path)-1):
                cur_node = ant_path[i]
                next_node = ant_path[i+1]

                delta_phers[cur_node][next_node] += ant_path_score
        
        phers *= p
        phers += delta_phers

        print(np.around(phers, 2))

if __name__ == '__main__':
    ant_sys()
    
