# Nosso Projeto da prática de PSO

Este aqui é o nosso projeto de **Particle Swarm Optimization (PSO)**, desenvolvido seguindo o que a gente viu nas aulas de computação inteligente lecionado pelo professor Élisson Rocha.


## O que tem no algoritmo?

A gente implementou as seguintes regras:

*   **Topologia Global:** Todo mundo no enxame sabe quem é o `gbest` (o melhor do grupo).
*   **Inércia Dinâmica ($w$):** Começamos em $0.9$ pra explorar o mapa todo e terminamos em $0.4$ pra focar no lugar certo.
*   **Coeficientes de Confiança:** Usamos $c1 = 2.05$ (o quanto a partícula confia nela mesma) e $c2 = 2.05$ (o quanto ela confia no grupo).
*   **Função Objetivo:** Por padrão, estamos usando a **Sphere**, que é basicamente a soma dos quadrados. Se quiser testar outra, o código tá fácil de mexer.

## O que você precisa?

Só do Python 3 e do **NumPy** instalado (pra fazer as contas pesadas pros nossos vetores):

```bash
pip install numpy
```

## Como rodar?

É só abrir o terminal na pasta do projeto e rodar:

```bash
python pso.py
```

## Estrutura do Código

*   `Particle`: A classe de cada partícula (posição, velocidade e memória).
*   `PSO`: Onde tudo acontece e o enxame é gerenciado.
*   `sphere_function`: A nossa função de teste.

Qualquer dúvida, dá uma olhada nos comentários do `pso.py` que a gente explicou o passo a passo de como a velocidade e a posição são atualizadas.