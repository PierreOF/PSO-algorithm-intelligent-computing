import numpy as np
from pso import PSO

# Funções benchmark clássicas pra testar o PSO
# Cada uma tem um mínimo global conhecido, então a gente consegue avaliar o quão perto o algoritmo chega

def sphere(x):
    """Mínimo global: f(0,...,0) = 0"""
    return np.sum(x**2)

def rastrigin(x):
    """Mínimo global: f(0,...,0) = 0. Muitos mínimos locais, difícil de otimizar."""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """Mínimo global: f(1,...,1) = 0. O 'vale' é plano e difícil de navegar."""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    """Mínimo global: f(0,...,0) = 0. Tem vários mínimos locais na borda."""
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e

def griewank(x):
    """Mínimo global: f(0,...,0) = 0. Muitos mínimos locais distribuídos."""
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1


# Configuração dos testes: função, limites, valor esperado do mínimo
testes = [
    {"nome": "Sphere",       "funcao": sphere,     "limites": (-10, 10),  "minimo_esperado": 0.0},
    {"nome": "Rastrigin",    "funcao": rastrigin,  "limites": (-5.12, 5.12), "minimo_esperado": 0.0},
    {"nome": "Rosenbrock",   "funcao": rosenbrock, "limites": (-5, 5),    "minimo_esperado": 0.0},
    {"nome": "Ackley",       "funcao": ackley,     "limites": (-32.768, 32.768), "minimo_esperado": 0.0},
    {"nome": "Griewank",     "funcao": griewank,   "limites": (-600, 600), "minimo_esperado": 0.0},
]

# Parâmetros fixos pro teste
DIMENSOES = 10
PARTICULAS = 30
ITERACOES = 200


def rodar_testes():
    print("=" * 70)
    print(f"  TESTE DO PSO — {len(testes)} funções benchmark")
    print(f"  Dimensões: {DIMENSOES} | Partículas: {PARTICULAS} | Iterações: {ITERACOES}")
    print("=" * 70)

    resultados = []

    for teste in testes:
        nome = teste["nome"]
        print(f"\n{'-' * 70}")
        print(f"  Funcao: {nome} (minimo esperado: {teste['minimo_esperado']})")
        print(f"{'-' * 70}")

        pso = PSO(
            objective_func=teste["funcao"],
            dim=DIMENSOES,
            bounds=teste["limites"],
            num_particles=PARTICULAS,
            max_iter=ITERACOES,
        )

        best_pos, best_fit = pso.optimize()
        erro = abs(best_fit - teste["minimo_esperado"])

        resultados.append({
            "nome": nome,
            "resultado": best_fit,
            "erro": erro,
            "posicao": best_pos,
        })

    # Resumo final
    print(f"\n{'=' * 70}")
    print("  RESUMO DOS RESULTADOS")
    print(f"{'=' * 70}")
    print(f"{'Função':<15} {'Resultado':>15} {'Erro':>15} {'Status':>10}")
    print(f"{'-' * 55}")

    for r in resultados:
        # Tolerância: pra 10 dimensões, consideramos bom se o erro for < 1.0
        status = "OK" if r["erro"] < 1.0 else "RUIM"
        print(f"{r['nome']:<15} {r['resultado']:>15.6e} {r['erro']:>15.6e} {status:>10}")

    print(f"{'-' * 55}")

    acertos = sum(1 for r in resultados if r["erro"] < 1.0)
    print(f"\n  {acertos}/{len(resultados)} funções com erro < 1.0")


if __name__ == "__main__":
    rodar_testes()
