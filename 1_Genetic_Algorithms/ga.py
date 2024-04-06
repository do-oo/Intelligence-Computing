import random
import numpy as np
import copy
from pprint import pprint

def eval_fun(x):
    f = np.sum(np.abs(x)) + 1
    return 1/f

def get_init_solution(N=10):
    xs = np.random.uniform(-10, 10, size=[N, 4])
    return xs

def get_sum_prob(scores):
    scores = np.array(scores)
    scores /= np.sum(scores)
    # print("score", scores)
    sum_probs = np.zeros_like(scores)

    sum_probs[0] = scores[0]
    for i in range(1, len(scores)):
        sum_probs[i] = sum_probs[i-1] + scores[i]
    return sum_probs

def selection(xs, scores):
    probs = get_sum_prob(scores)
    # print("probs", probs)
    N = len(xs)
    ys = []
    for i in range(N):
        for j in range(N):
            if random.uniform(0, 1) < probs[j]:
                ys.append(xs[j])
                break
    return ys


def crossover(xs, candidates_prob_cross_over=[0.49, 0.90]):
    prob = random.uniform(candidates_prob_cross_over[0], candidates_prob_cross_over[1])

    xs_probs = np.random.uniform(0, 1, len(xs))

    mask_cross_over = xs_probs < prob
    # print(mask_cross_over)
    indics = np.where(mask_cross_over)[0]
    
    if len(indics) < 2: return xs

    # N = len(xs[i])//2
    for i in range(1, len(indics), 2):
        j = i-1

        # x[i][:N]
        pos = np.random.choice(range(0, len(xs[i])), 1)[0]
        xs[i][pos], xs[j][pos] = xs[j][pos], xs[i][pos]
        # tmp = copy.deepcopy(xs[i][pos:])
        # # # print("c", tmp)
        # xs[i][pos:] = xs[j][pos:]
        # # print("c2", tmp)
        # xs[j][pos:] = tmp


    return xs

def mutation(xs, probs=[0.1, 0.9]):
    for x in xs:
        prob = random.uniform(probs[0], probs[1])
        for i in range(len(x)):
            if random.uniform(0, 1) < prob:

                x[i] = random.uniform(-10, 10)
                # print("mut")
    return xs

def my_print(xs):
    pass
    # for i,x in enumerate(xs):
    #     print(i, x)
 

def main():
    # 0 设定初始参数
    max_steps = 10
    gt = 1
    N = 10000

    # xs_prob = np.ones([N])/N
    

    # 得到初始种群
    xs = get_init_solution(N)

    best_xs_score = -1
    best_xs = None
    print("init")
    my_print(xs)
    for i in range(max_steps):
        # 1. 适应性评价
        # print(f"===================== step {i}=========================")
        xs_score = [eval_fun(x) for x in xs]
        # print(xs_score)
        max_index = np.argmax(xs_score)
        if xs_score[max_index] > best_xs_score:
            best_xs_score = xs_score[max_index]
            best_xs = xs[max_index]

            print(f"iter {i}", best_xs, best_xs_score)

        # 2. 根据适应性概率选择个体
        xs = selection(xs, xs_score)
        # print("select")
        my_print(xs)
        # 3. 开始交配，开始遗传
        xs = crossover(xs, [0.8, 1])
        # print("crossover")
        my_print(xs)
        # 4. 变异
        xs = mutation(xs, [0.01, 0.1])
        # print("mutation")
        my_print(xs)
    # print("best xs", xs_score, xs)
        

        


        
        



def test1():
    xs = np.array([1, 2], dtype=np.float64)

    probs = np.array([1, 2],dtype=np.float64)
    
    data = []
    for i in range(300):
        # 使用np.unique函数获取唯一值和它们的频率
        d = selection(xs, probs)
        data.extend(d)
        # d = random.uniform(0, 1)
        # if d<0.5:
        #     data.append(1)
        # else:
        #     data.append(2)
    # print(data)

    unique_values, frequencies = np.unique(data, return_counts=True)

    # 打印唯一值和频率
    for value, frequency in zip(unique_values, frequencies):
        print(f"值 {value} 的频率为 {frequency}")    

def test2():

    xs = np.array([(1, 3), (2, 4)], dtype=np.float64)
    print(xs)
    xs = crossover(xs, [0.9, 0.1])
    print(xs)

if __name__ == '__main__':
    main()
