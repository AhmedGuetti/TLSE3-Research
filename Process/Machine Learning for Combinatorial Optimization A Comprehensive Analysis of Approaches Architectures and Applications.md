## Abstract

This paper presents a comprehensive analysis of machine learning approaches for solving combinatorial optimization problems. We examine the evolution from traditional exact and heuristic methods to recent advances in neural combinatorial optimization, analyzing various architectural paradigms, learning strategies, and representation techniques. Our comparative analysis focuses on three primary domains: general combinatorial optimization problems including the Traveling Salesman Problem, Branch-and-Bound methods for Mixed Integer Linear Programming, and Boolean Satisfiability solvers. For each domain, we investigate the trade-offs between supervised learning via imitation and reinforcement learning approaches, and compare performance metrics, generalization capabilities, and computational efficiency across different architectures. Key insights are provided regarding the advantages and limitations of various graph neural network architectures, feature engineering approaches, and the critical interplay between traditional algorithmic frameworks and neural components. Our analysis demonstrates that while significant advances have been made, there remain substantial challenges in developing machine learning methods that can consistently outperform traditional algorithms across heterogeneous problem instances.

## 1. Introduction

Combinatorial optimization (CO) problems arise in numerous industrial and scientific domains, including logistics, scheduling, network design, and resource allocation. These problems are characterized by discrete decision variables and combinatorial solution spaces, making them computationally challenging. Traditional approaches to solving CO problems include exact methods, such as branch-and-bound and dynamic programming, and heuristic methods, such as local search and genetic algorithms. However, exact methods often become intractable for large problem instances, while heuristic methods lack performance guarantees and require careful parameter tuning.

In recent years, machine learning (ML) approaches, particularly deep learning, have shown promising results in tackling CO problems. These methods, collectively termed Neural Combinatorial Optimization (NCO), aim to leverage data-driven learning to improve solution quality, computational efficiency, or both. The integration of ML with CO algorithms has opened new research avenues, with various approaches being explored to address different problem classes.

This paper provides a systematic comparison of ML approaches for solving CO problems, focusing on three primary domains:

1. General CO problems, with emphasis on the Traveling Salesman Problem (TSP)
2. Branch-and-Bound methods for Mixed Integer Linear Programming (MILP)
3. Boolean Satisfiability (SAT) solvers

For each domain, we examine the architectural paradigms, learning strategies, representation techniques, and evaluation methodologies used in the literature. Our analysis extends beyond a traditional survey by providing a comparative framework to assess the relative strengths and limitations of different approaches.

The remainder of this paper is organized as follows: Section 2 presents the background and preliminaries on CO problems and machine learning approaches. Sections 3, 4, and 5 analyze ML approaches for general CO problems, MILP solvers, and SAT solvers, respectively. Section 6 provides a comparative analysis across domains, and Section 7 concludes with insights and future research directions.

## 2. Background and Preliminaries

### 2.1 Combinatorial Optimization Problems

A combinatorial optimization problem can be formally defined as finding a solution x∗x^* x∗ from a finite or countably infinite set of feasible solutions X\mathcal{X} X that minimizes an objective function f:X→Rf: \mathcal{X} \rightarrow \mathbb{R} f:X→R:

x∗=arg⁡min⁡x∈Xf(x)x^* = \arg\min_{x \in \mathcal{X}} f(x)x∗=argx∈Xmin​f(x)

The solution space X\mathcal{X} X is typically defined by a set of constraints, and the structure of this space varies across problem classes.

#### 2.1.1 Traveling Salesman Problem (TSP)

Given a set of cities and the distances between them, the TSP seeks to find the shortest possible route that visits each city exactly once and returns to the origin city. Formally, given a complete graph G=(V,E)G = (V, E) G=(V,E) with vertex set VV V representing cities and edge set EE E with weights representing distances, the goal is to find a Hamiltonian cycle of minimum total weight.

#### 2.1.2 Mixed Integer Linear Programming (MILP)

A MILP problem can be formulated as:

min⁡x{cTx:Ax≤b,x≥0,xj∈Z∀j∈I}\min_{x} \{c^T x : Ax \leq b, x \geq 0, x_j \in \mathbb{Z} \forall j \in I\}xmin​{cTx:Ax≤b,x≥0,xj​∈Z∀j∈I}

where c∈Rnc \in \mathbb{R}^n c∈Rn, A∈Rm×nA \in \mathbb{R}^{m \times n} A∈Rm×n, b∈Rmb \in \mathbb{R}^m b∈Rm, and I⊆{1,2,...,n}I \subseteq \{1, 2, ..., n\} I⊆{1,2,...,n} is the set of indices of variables that are constrained to be integers.

#### 2.1.3 Boolean Satisfiability Problem (SAT)

The SAT problem involves determining if there exists an assignment of truth values to a set of Boolean variables that makes a given Boolean formula evaluate to true. The formula is typically expressed in conjunctive normal form (CNF), consisting of a conjunction of clauses, where each clause is a disjunction of literals.

### 2.2 Machine Learning Approaches
#### 2.2.1 Supervised Learning from Demonstration

In supervised learning, a model is trained to imitate an expert or oracle by minimizing a loss function between the model's predictions and the expert's decisions:

min⁡θEs∼D[ℓ(πθ(s),πexpert(s))]\min_{\theta} \mathbb{E}_{s \sim \mathcal{D}} [\ell(\pi_\theta(s), \pi_\text{expert}(s))]θmin​Es∼D​[ℓ(πθ​(s),πexpert​(s))]

where πθ\pi_\theta πθ​ is the model's policy parameterized by θ\theta θ, πexpert\pi_\text{expert} πexpert​ is the expert policy, and D\mathcal{D} D is a distribution over states.

#### 2.2.2 Reinforcement Learning from Experience

In reinforcement learning (RL), a policy is learned through interactions with an environment, represented as a Markov Decision Process (MDP) $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$, where $\mathcal{S}$ is the state space, $\mathcal{A}$is the action space, $\mathcal{P}$ is the transition probability function, $\mathcal{R}$ is the reward function, and $\gamma$ is the discount factor. The goal is to find a policy $\pi^*$ that maximizes the expected cumulative discounted reward:

$\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t \mathcal{R}(s_t, a_t) \right]$


#### 2.2.3 Neural Network Architectures

Several neural network architectures have been proposed for CO problems:

1. **Pointer Networks (PN)**: Recurrent neural networks with attention mechanisms that can generate permutations of input sequences.
2. **Graph Neural Networks (GNN)**: Neural networks that operate on graph-structured data, capturing the relational structure of CO problems.
3. **Graph Attention Networks (GAT)**: GNNs that use attention mechanisms to weight the importance of different neighbors.
4. **Graph Convolutional Networks (GCN)**: GNNs that apply convolutional operations to graph-structured data.

## 3. Machine Learning for General Combinatorial Optimization Problems

### 3.1 Architectural Approaches

Recent research has explored various neural architectures for solving general CO problems, with particular emphasis on the TSP. These approaches can be categorized as follows:

#### 3.1.1 End-to-End Neural Solvers

Wang et al. highlight that deep neural networks (DNNs) have been increasingly employed to solve CO problems directly, without relying on traditional algorithmic frameworks. These end-to-end approaches leverage neural networks to map problem instances directly to solutions.

One of the pioneering works in this category is the Pointer Network (PN) architecture, which uses attention mechanisms to generate permutations of the input sequence. This architecture is particularly suitable for problems like TSP, where the solution is a permutation of the input cities.

More recent approaches have utilized Graph Neural Networks (GNNs) to better capture the graph structure inherent in many CO problems. GNNs operate by passing messages between nodes, allowing them to learn complex patterns and relationships within the graph.

#### 3.1.2 Hybrid Neural-Algorithmic Approaches

Chung et al. discuss the trend of combining neural networks with traditional algorithmic frameworks, particularly in industrial engineering contexts. These hybrid approaches aim to leverage the strengths of both paradigms: the adaptability and learning capabilities of neural networks, and the theoretical guarantees and efficiency of traditional algorithms.

In the context of routing problems like TSP, these hybrid approaches often involve using neural networks to guide search algorithms or to make decisions within a traditional algorithmic framework.

### 3.2 Learning Strategies

The literature reveals two primary learning strategies for CO problems:

#### 3.2.1 Supervised Learning via Imitation

This approach involves training neural networks to imitate expert solutions or heuristics. The network is trained on a dataset of problem instances and their corresponding optimal or high-quality solutions. The objective is to minimize the difference between the network's predictions and the expert solutions.

While this approach can be effective, it is limited by the quality of the available expert solutions and may struggle to generalize to problem instances significantly different from those in the training set.

#### 3.2.2 Reinforcement Learning via Experience

Mazyavkina et al. provide an extensive survey on the use of reinforcement learning for CO problems. In this approach, a neural network interacts with an environment (the CO problem) and learns from the rewards or penalties it receives based on the quality of its decisions.

RL approaches for CO problems typically model the solution process as a sequential decision-making problem, where at each step, the agent (neural network) selects an action (e.g., the next city to visit in TSP) based on the current state.

One of the advantages of RL over supervised learning is that it does not require expert solutions for training. However, RL can be more challenging to train due to issues like sparse rewards, exploration-exploitation trade-offs, and credit assignment.

### 3.3 Evaluation Metrics and Performance Comparisons

The performance of NCO methods is typically evaluated using several metrics:

1. **Solution Quality**: Measured by the gap between the solution found by the neural method and the optimal solution or the best-known solution.
2. **Computational Efficiency**: Measured by the time taken to find a solution, which is particularly important for real-time applications.
3. **Generalization**: Assessed by the ability of the method to perform well on problem instances different from those seen during training, including larger instances or instances from different distributions.

Mazyavkina et al. note that RL-based approaches have shown promising results in terms of computational efficiency compared to classical heuristic methods, making them increasingly relevant for real-world applications. However, they often still lag behind state-of-the-art exact methods in terms of solution quality for instances where exact solutions are tractable.

Wang et al. highlight that despite advancements, DNN-based methods for CO problems still have significant room for improvement, particularly in terms of feature extraction capabilities. The fact that these methods do not rely heavily on expert knowledge has contributed to their popularity in recent years.

## 4. Machine Learning for Branch-and-Bound in MILP

Branch-and-Bound (B&B) is a fundamental algorithmic framework for solving MILPs. The B&B algorithm recursively partitions the solution space by branching on variables and uses bounds to prune the search tree. Several research efforts have focused on enhancing B&B with machine learning techniques.

### 4.1 Branching Variable Selection

One of the critical decisions in B&B is selecting the variable to branch on at each node. Different approaches have been proposed to learn effective branching policies:

#### 4.1.1 Feature Engineering for Branching

Marcos Alvarez et al. propose a machine learning-based approximation of strong branching, a computationally expensive but effective branching strategy. Their approach involves training an Extremely Randomized Trees (ExtraTrees) model using carefully engineered features to represent the state of the B&B process.

The features include:

1. Static problem features derived from the problem parameters
2. Dynamic problem features related to the current B&B node
3. Dynamic optimization features capturing the history of the optimization process

This approach showed significant speedups compared to strong branching while maintaining solution quality, with a three-fold reduction in computation time for some problem instances.

#### 4.1.2 Tree State Parameterization

Zarpellon et al. introduce a novel approach for learning branching policies by parameterizing the state of the B&B search tree. They argue that the state of the search tree should condition the branching criteria to adapt to different stages of the optimization process.

Their approach involves two key components:

1. A representation of candidate variables for branching based on their roles in the search
2. A tree state representation that provides context for branching decisions

They propose two neural network architectures:

1. NoTree: A baseline architecture that processes candidate variables without tree context
2. TreeGate: An architecture that incorporates the tree state to modulate the candidate variable representations via feature gating

The TreeGate architecture showed a 19% improvement in test accuracy compared to NoTree, highlighting the importance of incorporating search tree information in branching decisions.

### 4.2 Learning Strategies for B&B

#### 4.2.1 Imitation Learning

Both Marcos Alvarez et al. and Zarpellon et al. use imitation learning to train their branching policies. In this approach, the policy is trained to mimic the decisions of an expert branching strategy (often strong branching or a variant thereof).

Zarpellon et al. use SCIP's default branching rule (relpscost) as their expert policy. The training data consists of pairs of input features (representing the state of the B&B process) and target branching decisions.

#### 4.2.2 Transfer Learning and Generalization

A critical challenge in learning branching policies is ensuring generalization to unseen problem instances. Zarpellon et al. explicitly target generalization across heterogeneous MILP instances, i.e., problems from different domains with varying structures and sizes.

Their results demonstrate that the TreeGate architecture enables better generalization compared to the NoTree architecture, with a 27% reduction in the number of nodes explored for test instances.

### 4.3 Integration with Solver Infrastructure

Scavuzzo et al. discuss the challenges of integrating ML components into MILP solvers, particularly the interaction between CPU-based solvers and GPU-accelerated ML models. They highlight the potential of leveraging solver statistics to adapt optimization approaches automatically, presenting new research opportunities at the intersection of ML and mathematical optimization.

The authors also distinguish between approaches that specialize in specific problem structures and those that aim to generalize across heterogeneous instances, noting the trade-offs involved in this choice.

## 5. Machine Learning for SAT Solvers

The Boolean Satisfiability Problem (SAT) is a fundamental problem in computer science with applications in formal verification, planning, and scheduling. Traditional SAT solvers include complete methods like Conflict-Driven Clause Learning (CDCL) and incomplete methods like stochastic local search (SLS).

### 5.1 End-to-End Neural SAT Solvers

Guo et al. discuss the evolution of ML-based SAT solvers from naive classifiers with handcrafted features to end-to-end solvers. They highlight how SAT instances can be encoded as graphs, with literals and clauses as nodes, and how Graph Neural Networks (GNNs) can be trained to solve these instances.

A notable example is NeuroSAT, which uses a message-passing neural network to predict satisfiability. NeuroSAT shows promising results on random SAT instances but has limitations in terms of scalability and generalization to larger instances.

### 5.2 Enhancing Traditional SAT Solvers with ML

#### 5.2.1 Branching Heuristics in CDCL Solvers

Similar to B&B for MILP, CDCL solvers for SAT require making branching decisions, i.e., selecting variables to assign truth values to. Guo et al. discuss several approaches to learn branching heuristics for SAT solvers, including:

1. NeuroCore: A GNN-based model that periodically updates the activity scores used by the VSIDS branching heuristic
2. Graph-Q-SAT: A reinforcement learning approach that uses GNNs to learn a branching policy

#### 5.2.2 Learning from Unsatisfiability

A unique aspect of SAT solving is the importance of unsatisfiability (UNSAT) proofs. Guo et al. mention approaches like NeuroCuber, which learns to predict the occurrence of variables in DRAT proofs, helping to minimize the size of resolution trees for UNSAT instances.

### 5.3 Reinforcement Learning for SAT

Kurin et al. investigate the use of Q-learning with GNNs to learn a generalizable branching heuristic for SAT solvers. Their approach, Graph-Q-SAT, represents SAT instances as variable-constraint graphs and learns a value function for each variable through reinforcement learning.

Key components of their approach include:

1. A state representation that encodes the current state of the SAT solver
2. A reward function that encourages early pruning of the search tree
3. A GNN architecture that can process graphs of varying sizes

Their results show that Graph-Q-SAT can achieve a 2-3X reduction in iterations for problems up to 5X larger than those seen during training, and a 1.5-2X reduction when generalizing from SAT to UNSAT instances.

However, they also note limitations, particularly in terms of wall-clock time reduction, which makes traditional heuristic methods more adaptable for industrial applications.

## 6. Comparative Analysis

Having examined ML approaches for different CO domains, we now provide a comparative analysis across these domains, focusing on architectural paradigms, learning strategies, and empirical performance.

### 6.1 Architectural Paradigms

#### 6.1.1 Graph Neural Networks

GNNs emerge as a dominant architectural paradigm across all three domains. Their ability to process graph-structured data makes them particularly suitable for CO problems, which often have an inherent graph structure.

In general CO problems, GNNs are used to capture the structure of the problem instance, such as the distances between cities in TSP. In MILP, GNNs process the bipartite graph representing the constraint matrix. In SAT, GNNs operate on literal-clause graphs or variable-constraint graphs.

The specific GNN architectures vary across domains, with Graph Convolutional Networks (GCNs) being common in MILP applications and more general message-passing neural networks being used in SAT solvers.

#### 6.1.2 Recurrent Neural Networks

RNNs, particularly with attention mechanisms like Pointer Networks, are more common in general CO problems where the solution is a sequence or permutation, such as TSP. They are less prevalent in MILP and SAT applications, where the solution structure is more complex.

#### 6.1.3 Hybrid Architectures

Hybrid architectures that combine neural components with traditional algorithmic frameworks are increasingly common across all domains. These approaches aim to leverage the strengths of both paradigms: the adaptability and learning capabilities of neural networks, and the theoretical guarantees and efficiency of traditional algorithms.

In MILP, Marcos Alvarez et al. and Zarpellon et al. integrate neural models into the B&B framework to guide branching decisions. In SAT, Graph-Q-SAT incorporates a GNN into the CDCL solver to learn a branching policy. These hybrid approaches have shown promising results in terms of balancing solution quality and computational efficiency.

### 6.2 Learning Strategies

#### 6.2.1 Supervised vs. Reinforcement Learning

A clear division exists between approaches that use supervised learning via imitation and those that use reinforcement learning via experience.

Supervised learning is more common in MILP applications, where the expert policy (often strong branching) provides a clear target for imitation. Zarpellon et al. and Marcos Alvarez et al. both use imitation learning to train their branching policies.

Reinforcement learning is more prevalent in general CO problems and SAT solvers, where the solution process can be more naturally framed as a sequential decision-making problem. Kurin et al. and Mazyavkina et al. highlight the use of RL for learning branching policies in SAT and general CO problems, respectively.

#### 6.2.2 Data Requirements

The data requirements differ significantly between supervised and reinforcement learning approaches. Supervised learning requires a dataset of expert decisions, which can be expensive to generate, particularly for large problem instances. For example, generating strong branching decisions for MILP instances requires solving many LP relaxations.

Reinforcement learning does not require expert demonstrations but instead learns from interactions with the environment. This can be more data-efficient but often requires careful reward engineering and may suffer from exploration challenges.

### 6.3 Empirical Performance

#### 6.3.1 Solution Quality vs. Computational Efficiency

Across all domains, there is a trade-off between solution quality and computational efficiency. Traditional exact methods typically provide optimal solutions but are computationally expensive, while heuristic methods are faster but may not find optimal solutions.

ML approaches aim to bridge this gap by providing high-quality solutions with reduced computational cost. Marcos Alvarez et al. report a three-fold reduction in computation time compared to strong branching for some MILP instances, while maintaining solution quality. Similarly, Kurin et al. demonstrate a 2-3X reduction in iterations for SAT problems.

#### 6.3.2 Generalization Capabilities

Generalization to unseen problem instances remains a significant challenge across all domains. Zarpellon et al. explicitly target this issue in MILP, showing that their TreeGate architecture enables better generalization across heterogeneous instances. Kurin et al. also address generalization in SAT, demonstrating that Graph-Q-SAT can generalize to larger instances and from SAT to UNSAT problems.

However, Wang et al. note that DNN-based methods for general CO problems still have limitations in terms of feature extraction capabilities, which affects their generalization performance.

#### 6.3.3 Scalability

Scalability to larger problem instances is another common challenge. Guo et al. highlight that end-to-end neural SAT solvers like NeuroSAT struggle to scale to larger instances. Similarly, Wang et al. note that ML approaches for general CO problems have limitations in handling large instances.

In MILP, Scavuzzo et al. discuss the challenges of integrating ML components into solvers, particularly the computational overhead of neural networks, which can offset the benefits of improved decision-making.

## 7. Conclusion and Future Directions

This paper has provided a comprehensive comparative analysis of machine learning approaches for combinatorial optimization problems. We have examined architectural paradigms, learning strategies, and empirical performance across three domains: general CO problems (with emphasis on TSP), MILP solvers, and SAT solvers.

Several key insights emerge from our analysis:

1. **Graph Neural Networks** have emerged as a dominant architectural paradigm across all domains, thanks to their ability to process graph-structured data and capture the relational nature of CO problems.
2. **Hybrid neural-algorithmic approaches** that combine ML components with traditional algorithmic frameworks show the most promise in terms of balancing solution quality and computational efficiency.
3. **Learning strategies** vary across domains, with supervised learning via imitation being more common in MILP applications and reinforcement learning being more prevalent in general CO problems and SAT solvers.
4. **Generalization** across heterogeneous problem instances remains a significant challenge, with approaches like tree state parameterization in MILP showing promise in addressing this issue.
5. **Scalability** to larger problem instances is another common challenge, with computational overhead of neural networks potentially offsetting the benefits of improved decision-making.

Future research directions include:

1. **Developing more efficient neural architectures** that can process large problem instances without excessive computational overhead.
2. **Exploring hybrid learning strategies** that combine the benefits of supervised and reinforcement learning, such as using supervised learning for initialization followed by reinforcement learning for refinement.
3. **Investigating transfer learning and meta-learning** approaches to improve generalization across different problem distributions.
4. **Addressing the integration challenges** of incorporating ML components into traditional solver infrastructure, particularly the CPU-GPU interaction.
5. **Developing standardized benchmarks and evaluation methodologies** to enable fair comparisons across different approaches.

Machine learning for combinatorial optimization is a rapidly evolving field with significant potential for impact across various domains. While substantial progress has been made, there remain ample opportunities for further research and innovation.

## References

1. Chung, Lee, & Tsang. Neural combinatorial optimization with reinforcement learning in industrial engineering: A survey.
2. Mazyavkina, Sviridov, Ivanov, & Burnaev. Reinforcement learning for combinatorial optimization: A survey.
3. Wang, He, & Li. Solving Combinatorial Optimization Problems with Deep Neural Network: A Survey.
4. Marcos Alvarez, A., Louveaux, Q., & Wehenkel, L. (2017). A Machine Learning-Based Approximation of Strong Branching.
5. Zarpellon, G., Jo, J., Lodi, A., & Bengio, Y. (2021). Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies.
6. Scavuzzo, L., Aardal, K., Lodi, A., & Yorke-Smith, N. Machine Learning Augmented Branch and Bound for Mixed Integer Linear Programming.
7. Guo, W., Yan, J., Zhen, H. L., Li, X., Yuan, M., & Jin, Y. Machine Learning Methods in Solving the Boolean Satisfiability Problem.
8. Kurin, V., Godil, S., Whiteson, S., & Catanzaro, B. Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver?
9. Bengio, Y., Lodi, A., & Prouvost, A. (2020). Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon. European Journal of Operations Research.