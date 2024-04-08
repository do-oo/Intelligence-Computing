import random
import numpy as np

n_particle = 3
epoch = 10
pos_bound = [[-10, 10], [-10, 10]]

class Particle():
    def __init__(self, pos_bound, fitness) -> None:
        self.bound = np.array(pos_bound)
        self.fitness = fitness

        self.cur_pos = np.array([random.uniform(*bound) for bound in pos_bound])
        self.cur_v = np.array([random.uniform(*bound) for bound in pos_bound])* random.uniform(-0.05, 0.05)

        self.p_best_pos = self.cur_pos
        self.p_best_fitness = fitness(self.cur_pos)

        self.c1 = 2
        self.c2 = 2
        self.w = 0.9
        

    def __repr__(self) -> str:
        p = f"cur_pos {self.cur_pos}, cur_v {self.cur_v}, p_best {self.p_best_pos}, p_best_fiteness {self.p_best_fitness}"
        return p
    
    def update(self, g_best):
        self.cur_v = self.w * self.cur_v + \
            self.c1 * random.uniform(0, 1) * (self.p_best_pos-self.cur_pos) + \
            self.c2 * random.uniform(0, 1) * (g_best-self.cur_pos)
        
        self.cur_v = np.clip(self.cur_v, self.bound[:, 0]*0.1, self.bound[:, 1]*0.1)
        
        pos = self.cur_pos + self.cur_v
        
        self.cur_pos = np.clip(pos, self.bound[:, 0], self.bound[:, 1])

        cur_fitness = self.fitness(self.cur_pos)

        if cur_fitness > self.p_best_fitness:
            self.p_best_fitness = cur_fitness
            self.p_best_pos = self.cur_pos

        self.w *= 0.999
        self.w = max(self.w, 0.4)


class Group():
    def __init__(self, n_particle, pos_bound) -> None:
        self.n_particle = n_particle
        self.n_pos_bound = pos_bound
        self.particles = [Particle(pos_bound, self.fitness) for i in range(self.n_particle)]
        self.update_best_fitness()

    @staticmethod
    def fitness(data):
        return  -(data[1] ** 2 + data[0] ** 2)
    
    def update_best_fitness(self):
        best_fitness = -float('inf') if not hasattr(self, "best_fitness") else getattr(self, "best_fitness")
        best_pos = None if not hasattr(self, "best_pos") else self.best_pos
        for i in range(self.n_particle):
            particle = self.particles[i]
            if particle.p_best_fitness > best_fitness:
                best_fitness = particle.p_best_fitness
                best_pos = particle.p_best_pos
        self.best_fitness = best_fitness
        self.best_pos = best_pos


    def __repr__(self) -> str:
        p = f"best_pos: {self.best_pos}, best_fitness: {self.best_fitness}"
        return p

    def run(self):
        for e in range(10):
            print(f"============{e} ==========")
            for i in range(self.n_particle):
                self.particles[i].update(self.best_pos)
                # print(self.particles[i])
            self.update_best_fitness()

            print(self)


if __name__ == '__main__':
    g = Group(10, pos_bound)
    g.run()