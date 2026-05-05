import numpy as np

# Definindo a função Sphere pra gente testar o algoritmo
# Basicamente ela soma os quadrados de cada dimensão. O objetivo é chegar em zero.
def sphere_function(x):
    return np.sum(x**2)

class Particle:
    """
    Aqui a gente cria a classe da Partícula. 
    Cada uma tem sua posição, velocidade e lembra do seu melhor resultado (pbest).
    """
    def __init__(self, bounds, dim):
        # A gente começa a partícula em um lugar aleatório dentro dos limites do problema
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        
        # A velocidade também começa aleatória pra ela já sair 'andando'
        self.velocity = np.random.uniform(-1, 1, dim)
        
        # No começo, a melhor posição dela é a posição onde ela nasceu
        self.pbest_pos = np.copy(self.position)
        self.pbest_fitness = float('inf') # Começamos com infinito pra qualquer valor ser melhor que esse
        
        self.fitness = float('inf')

    def evaluate(self, objective_func):
        # Aqui a gente calcula o quão boa está a posição atual
        self.fitness = objective_func(self.position)
        
        # Se o resultado atual for melhor que o que ela já teve antes, a gente atualiza o pbest
        if self.fitness < self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_pos = np.copy(self.position)

class PSO:
    """
    Essa classe gerencia o enxame todo e controla as iterações.
    """
    def __init__(self, objective_func, dim, bounds, num_particles, max_iter):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        # Parâmetros que o professor passou na aula da UPE
        self.c1 = 2.05  # Coeficiente cognitivo (o quanto a partícula confia nela mesma)
        self.c2 = 2.05  # Coeficiente social (o quanto ela segue o grupo)
        
        # Valores de inércia pro decaimento linear
        self.w_max = 0.9
        self.w_min = 0.4
        
        # Criando a nossa lista de partículas
        self.swarm = [Particle(bounds, dim) for _ in range(num_particles)]
        
        # O gbest guarda a melhor posição que qualquer partícula do grupo já encontrou
        self.gbest_pos = np.zeros(dim)
        self.gbest_fitness = float('inf')

    def optimize(self):
        # Loop principal das gerações/iterações
        for i in range(self.max_iter):
            
            # Calculando o decaimento linear do w (inércia)
            # Ele começa alto pra explorar tudo e vai diminuindo pra focar onde está o mínimo
            w = self.w_max - (i / self.max_iter) * (self.w_max - self.w_min)
            
            for particle in self.swarm:
                # Primeiro a gente vê onde cada partícula está
                particle.evaluate(self.objective_func)
                
                # Se alguém achou um lugar melhor que o gbest atual, atualizamos o gbest
                if particle.fitness < self.gbest_fitness:
                    self.gbest_fitness = particle.fitness
                    self.gbest_pos = np.copy(particle.position)
            
            # Agora a gente atualiza a velocidade e a posição de todo mundo
            for particle in self.swarm:
                # r1 e r2 dão aquela variada aleatória no movimento
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                # A famosa fórmula da velocidade: inércia + parte cognitiva + parte social
                cognitive_component = self.c1 * r1 * (particle.pbest_pos - particle.position)
                social_component = self.c2 * r2 * (self.gbest_pos - particle.position)
                
                particle.velocity = (w * particle.velocity) + cognitive_component + social_component
                
                # Atualizando a posição somando a velocidade nova
                particle.position = particle.position + particle.velocity
                
                # Importante: não deixa a partícula fugir dos limites que a gente definiu
                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

            # Mostrando o progresso pra gente acompanhar
            print(f"Iteração {i+1}/{self.max_iter} | Inércia w: {w:.4f} | Melhor gbest: {self.gbest_fitness:.6e}")

        return self.gbest_pos, self.gbest_fitness

if __name__ == "__main__":
    # Definindo os parâmetros pra rodar o teste
    DIMENSOES = 10
    LIMITES = (-10, 10)
    PARTICULAS = 30
    ITERACOES = 100

    print("--- Rodando o PSO (Algoritmo de Enxame de Partículas) ---")
    
    # Criando o objeto da otimização
    pso = PSO(
        objective_func=sphere_function,
        dim=DIMENSOES,
        bounds=LIMITES,
        num_particles=PARTICULAS,
        max_iter=ITERACOES
    )
    
    # Começando a busca
    best_pos, best_fit = pso.optimize()
    
    print("\n--- Fim da execução ---")
    print(f"O melhor valor que a gente conseguiu foi: {best_fit:.6e}")
