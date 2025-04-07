# Abstract 
This paper reviews the recent literature on leveraging machine learning to solve combinatorial optimization problems. We watch the improvement in the performance between the exact solution and heuristic methods to recent advances in neural combinatorial optimization. Our work consist of analyzing various architectural paradigms, such as the learning strategies, and representation techniques. This comparative analysis highlight three main domains: general combinatorial optimization problems including the Traveling Salesman Problem, Branch-and-Bound methods for Mixed Integer Linear Programming, and Boolean Satisfiability solvers. For each domain, we investigate the advantage and down side of the two technique supervised learning via imitation and reinforcement learning approaches, and compare the different performant across different architectures. Our work demonstrates that while significant advances have been made, we still have  challenges in developing machine learning methods that can outperform traditional algorithms across heterogeneous problems.

# Introduction
Combinatorial optimization Problems (COP) refers to a class of discrete optimization problems. COP search a set of discrete variable that cause the objective function to be minimal under some constrains, COP arise in different industrial and scientific domains, including logistics, transportation and communication moreover in financial portfolios .

Classic examples of the COP are  the Capacitated Vehicle Routing Problem (CVRP), Minimum Vertex Cover (MVC), Job+Scheduling Problem (JSP), and Traveling Salesman Problem (TSP), are all NP-hard, witch mean we can not find a polynomial-time algorithms to get the optimal solution of the problems introduces 

Current method to solve COP are relying on exact or approximation algorithms, using dynamic, heuristic or bound and bound solution to solve our problem. Yet the exact method have a big flow where it becomes intractable for large problem instances, while heuristic methods lack performance guarantees and require special parameter engineering.

In recent year, more and more paper are exploring the use of machine learning (ML) approaches to solve COP in a more efficient way, particularly the using of deep learning, that have shown promising results solving those problems. Those method are grouped under the name Neural Combinatorial Optimization (NCO), leveraging multiple policies and technique to improve the solution quality in a reasonable time frame,  multiple approaches have been explored to solve different problem classes.

-  Exact algorithms, those methods have a big computation constrain where they cannot give a solution in an explainable time it the computation power is not important,  we use enumeration, cutting plane, and branch-and-bound algorithms to get the solution using mathematical methods.
- Approximate algorithms, those method have a trad-off between accuracy and performance where we get a solution close to the optimal solution by a factor  yet reducing the number of iteration, this solution exploit the greedy, relaxation, and dynamic programming algorithms.
- Heuristic algorithms, this approach try to find a near optimal solution while keeping the time reasonable, it use  local search, ant colony optimization, particle swarm optimization, and simulated annealing algorithms,

This paper explore three primary domain in a systematic comparison of ML approaches for solving CO problems:

1. General CO problems, with emphasis on the Traveling Salesman Problem (TSP)
2. Branch-and-Bound methods for Mixed Integer Linear Programming (MILP)
3. Boolean Satisfiability (SAT) solvers


For each domain, We explore the architectural of the solution proposed and the learning strategies representation techniques, This paper extends beyond a typical survey by giving a comparative  framework to examine the strengths and limitations of different approaches.

## 2. Background and Preliminaries

### 2.1 Combinatorial Optimization Problems
Without loss of generality, a combinatorial optimization problem can be defined as a min-optimization program. where the descript variables describe the decision to be made, upon a set of imposed constrain, we have also an objective function usually to be minimized,    defines the measure of the quality of every feasible assignment of values to variables. We call this a linear program if we have the objective and the constrains are linear, moreover if we restrict the variables to be integer values, then we have a MILP problem showed in mixed-integer linear programming [2.1.2] 

Input: A finite set $\mathcal{N}$ with element costs $c : \mathcal{N} → \mathcal{R}$ and a collection $\mathcal{F}$ of “feasible” subsets of $\mathcal{N}$. 
Goal: $\min\Big\{\sum_{j \in S} c_j : S \in \mathcal{F}\Big\}.$
#### 2.1.1 Traveling Salesman Problem (TSP)

(In word) we have a set of $n$ cities and the distances between each one of them, the salesman want seek to visit each city once and only once and return to the origin city in the last iteration. The cost for to travel from city $i$ to $j$ is known is advance we denote it $c_{ij}$. The objective is to find this series of cities  that minimize the cost.

Formally, we represent the problem in a complete graph $G=(\mathcal{V},\mathcal{E})$ that can be represented by a simple matrix, with vertex set $\mathcal{V}$ representing the cities and the edge set $\mathcal{E}$ representing the cost or the distance between those cities in an Euclidian TSP, the goal is to find a Hamiltonian cycle of minimum total weight.


#### 2.1.2 Mixed Integer Linear Programming (MILP)
Mixed Integer Linear Programming (MILP) provides a powerful state-of-art to be solved due to more than 50 years of research, and it is used to solve a variety of problem in business and engineering, however one must be careful when working with MILP where it's clear that the complexity of MILP is associated with the integrality requirement on (some) variables witch make the feasible region on such problem nonconvex. in the other hand dropping the integrality requirement defines a proper relaxation of MILP (i.e., an optimization problem whose feasible region contains the MILP feasible region), witch happen to be an LP witch is polynomial solvable, making it easy to see that solving such problem require the use a branch-and-bound (B&B) algorithm to divide the feasible region on multiple LP problem and iteration on the tree [bb]. 

In this paper we explore the proposed technic  to enhance the B&B, where it is the method all commercial MILP solver use to solve such problem, 

Most commercial MILP solver use the B&B technic, so enhancing this approach is in constant development, in this paper we explore those new method used to enhance the performance and the accuracy of such technic.



`\begin{align} \min \text{ (or max) } cx \\ Ax \geq b & \quad (a) \\ x \in \mathbb{R}^n, \; x_j \in \mathbb{Z} \text{ for } j \in J & \quad (b) \end{align}`

#### 2.1.3 Boolean Satisfiability Problem (SAT)
The Boolean satisfiability (SAT) is known to be NP-complete problems involves determining whether it exists an assignment of truth values to a set of Boolean variable making a propositional logic formula
satisfied, a propositional formula is a Boolean expression, those expression are build upon Boolean variable and ANDs, ORs and negations operations. The formula is written in conjunctive normal form (CNF) most of the time, consisting of a conjunction of clauses, where each clause is a disjunction of literals.
SAT problems are important  for both industry and academia that impacts various fields.as result modern SAT solver are optimized and work very efferently, most commercial SAT solver use heuristic to speed up the result, those heuristic are built using expert domain knowledge and most of the time are corrected by try and error, yet architecture of SAT solver are open-source.

### 2.2 Machine Learning Approaches
#### 2.2.1 Supervised Learning from Demonstration
Supervised learning in machine learning, is the process of training a model to output a prediction by minimizing a loss function between the prediction and the expert's decisions, this type of training require a set of pair (feature/target) as input where the loss function at the end should approximate the target for each feature.

Given a set of input/target pairs $D={(xi,yi)}_{i=1}^N​$ from a joint distribution $P(\mathcal{X},\mathcal{Y})$, supervised learning aims to find a function $f_\theta:\mathcal{X}→\mathcal{Y}$ parametrized by $\theta$ that minimizes the expected loss:
$$
\min_{\theta \in \mathbb{R}^p} \mathbb{E}_{X,Y \sim P} \left[ \ell(Y, f_{\theta}(X)) \right]
$$
Since the true distribution $P$ is unknown, we can approximate this using some probability distribution over our training dataset.

#### 2.2.2 Reinforcement Learning from Experience
reinforcement learning (RL), is a machine learning paradigm where a policy have to be learned through interactions with an environment and getting a score based on the decision made.
We can represent it as a Markov Decision Process (MDP)  $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$,  where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $\mathcal{P}$ is the transition probability function, $\mathcal{R}$ is the reward function, and $\gamma$ is the discount factor. When running the RL we want to find a policy  $\pi^*$ that maximized the rewards

$\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t \mathcal{R}(s_t, a_t) \right]$

#### 2.2.3 Neural Network Architectures

Several neural network architectures have been proposed for CO problems:

1. **Pointer Networks (PN)**: Recurrent neural networks with attention mechanisms that can generate permutations of input sequences.
2. **Graph Neural Networks (GNN)**: Neural networks that operate on graph-structured data, capturing the relational structure of CO problems.


## 3. Machine Learning for General Combinatorial Optimization Problems

### 3.1 Architectural Approaches

Research has explored numerous architectures to solve CO problems using ML, in our work we focused TSP and SAT problems as classical example in the universe of NP-hard problems, we present in this section different architecture that were proposed to solve this kind of problem.

#### 3.1.1 End-to-End Neural Solvers
deep neural networks (DNNs) have shown promising result on solving CO problems, for that reason we can see more an increase in the use of DNNs in CO solver as highlighted in \cite{wang2024solving}, without the use of traditional algorithmic. These end-to-end approaches leverage neural networks to map problem instances directly to solutions.


As discussed in section 223 pointer network is one of the most used architecture for ML solution sue to it attention mechanism to generate permutations of the input sequence. witch adapt perfectly with problems like TSP, where the solution is a permutation of the input cities  

in more recent work Graph Neural Networks (GNNs) architecture are being used to capture graph structure allowing the network to lean complex patterns and relationships within the graph due to it ability to pass messages between nodes in the network.

#### 3.1.2 Hybrid Neural-Algorithmic Approaches
\cite{chung2025neural} discussed the important of using CNNs with traditional algorithmic frameworks, in the context of  industrial engineering. These hybrid approaches aim to get the best of the two world, the speed of an heuristic and the accuracy of exact algorithm leveraging the power of learning capability in ML and theoretical guarantees from the traditional approaches led to introducing those hybrid methods.

Innovation in this section stayed sharp by researcher trying to optimize B&B algorithm as descripted in  \cite{alvarez2017machine}, by making ML handle the chose of the brunching policy to get a faster result with the accuracy of B&B.

In the context of routing problems like TSP, these hybrid approaches often involve using neural networks to guide search algorithms or to make decisions within a traditional algorithmic framework.



### 3.2 Learning Strategies

The literature reveals two primary learning strategies for CO problems:

#### 3.2.1 Supervised Learning via Imitation

This approach have the goal of imitation expert solution or heuristics. The network is trained on a corpus of data that consist of instances of the problem and their optimal or high-quality solution generated by traditional algorithm. We aim to minimize the difference between the network's predictions and the expert solutions. Despite the effectiveness of this approach compared to other solution, it is clean that the limitation on training data and their quality may lead to underfitting on the system, therefore struggling to generalize to problem instances significantly different from those in the training set.


#### 3.2.2 Reinforcement Learning via Experience

Mazyavkina et al. \cite{} explains in his survey the potential of reinforcement learning in solving CO, a neural network interact with the environment and learn from the rewards function generated by the policy based on the quality of the decisions been made.

RL approach to solve CO problems modle the solution as a sequential decision-making problem, where at each step, the agent will selects an action based on the current state.

RL approach doesn't need expert solutions for training witch make it better than supervised learning. Nevertheless it still hard to train those type of algorithm  due to issues like sparse rewards, exploration-exploitation trade-offs, and credit assignment.
### 3.3 Evaluation Metrics and Performance Comparisons

The performance of NCO methods is typically evaluated using several metrics:

1. **Solution Quality**: Measured by the gap between the solution found by the neural method and the optimal solution or the best-known solution.
2. **Computational Efficiency**: Measured by the time taken to find a solution, which is particularly important for real-time applications.
3. **Generalization**: Assessed by the ability of the method to perform well on problem instances different from those seen during training, including larger instances or instances from different distributions.

Mazyavkina et al. \cite{mazyavkina2021reinforcement} investigate the important of RL in the advancment of solving CO problems, they have shown promosing result compared to heuristic methods in term of efficiency, making them sutable for real-time application were speed is a must, But in the other hand, they often lag behind exact methods in term of accuracy and the quality of the solution.

Wang et al. \cite{wang2024solving} hightlight that  despite all those advancement, DNN-bsed solution still have big room for improvement, especilly in term of feature extraction capability. The fact that there method don't rely on expert knowledge, has made them very popular in the resent yea, where different state-of-art has been proposed.





## 4. Machine Learning for Branch-and-Bound in MILP

Branch-and-bound (B&B) is a classical algorithm to solve MILPs. it recursively partitions the solutions space by choosing some variables and brunch on them and uses bounds to prune the search tree. enhancing this method was one of the most researched topic in the world of optimizing CO solution, where it is used in multiple industries and logistic real-life problem. 


### 4.1 Branching Variable Selection

One of the critical decisions in B&B is selecting the variable to branch on at each node. Different approaches have been proposed to learn effective branching policies:

#### 4.1.1 Feature Engineering for Branching
Marcos Alvarez et al.  \cite{alvarez2017machine} explore the potential of using a machine learning-based approximation to imitate strong branching, a computationally expensive but effective branching strategy. they proposed the use of an Extremely Randomized Trees (ExtraTrees) model with carefully engineered features to represent the state of the state of the B&B process.

They identify three essential properties these features should have:
1. **Size Independence**: Features must be independent of the problem instance size, allowing a single learned branching strategy to work across different problem sizes.
2. **Invariance to Irrelevant Changes**: Features should be invariant to changes like row/column permutations.
3. **Scale Independence**: Features should remain identical if problem parameters are multiplied by some factor. 

This hybrid method showed a promising result where it have a fine ~91% gap from the Strong branching algorithm while having a more than 80% faster result. witch is not a bad trade off. Moreover this paper shows that this new method can explore more node than the strong brunching if no time limitation is set, after the test of using the CPLEX's cuts we outperform all the other method by stunning x3 more speed, all the result are tested on the MILPLIB benchmark and compared against :
- **random branching**
- **most-infeasible**
- **nonchimerical**
- **full strong branching**
- **reliability branching**


The main problem with this method is that it's more optimized with problems that are similar to the training data witch lead performance varies across problem types so for that it run poorly on problem that are not well structured in the training data, witch lead to a conclusion that this method  perform better when the feature are well, representing the problem in hand, in the other the engineering of the feature should be done in a very careful way to represent the main aspect of the problem in hand. 
#### 4.1.2 Tree State Parameterization
Zarpellon et al. \cite{zarpellon2021parameterizing} show an implementation using ML to impose branch variable selection (BVS) in MILP solvers, therefore enhancing the branch and bound algorithm, where a single poor decision can lead in  significantly increase the search tree size (nodes), they introduces a DNN architecture to learn those decision using Imitation Learning, encoding the data in a $Tree_t$  parameterization of the state of the  B&B search tree. They argue that the state of the search tree should condition the branching criteria to adapt to different stages of the optimization process.

Their approach involves two key components:

1. A representation of candidate variables for branching based on their roles in the search
2. A tree state representation that provides context for branching decisions

They propose two neural network architectures:

1. NoTree: A baseline architecture that processes candidate variables without tree context
2. TreeGate: An architecture that incorporates the tree state to modulate the candidate variable representations via feature gating

The TreeGate architecture showed a 19% improvement in test accuracy compared to NoTree, highlighting the importance of incorporating search tree information in branching decisions.

### 4.2 Learning Strategies for B&B

#### 4.2.1 Imitation Learning

Both Marcos Alvarez et al. \cite{alvarez2017machine}  and Zarpellon et al. \cite{zarpellon2021parameterizing} use imitation learning to train their branching policies. The policies are trained to mimic a decision that an  expert branching strategy  would take we use most strong branching policies for more accurate result or a variant of it.

Zarpellon et al. \cite{zarpellon2021parameterizing} use the default SCIP's branching rule (relpscost) as their expert policy. The training data is made of pairs of input features (representing the state of the B&B process) and target branching decisions.

#### 4.2.2 Transfer Learning and Generalization
One big limitation of learning branching policies is ensuring a generalization to unseen world of problems instances. Zarpellon et al. \cite{zarpellon2021parameterizing} explicitly target generalization across heterogeneous MILP instances, i.e., problems from different domains with varying structures and sizes.

Their TreeGate architecture shown better result on more generalized problems  compared to the NoTree architecture, with 27\% reduction of number of nodes explored for test instances.


### 4.3 Integration with Solver Infrastructure

Scavuzzo et al. \cite{scavuzzo2023machine} explore the challenge of integrating machine learning (ML) components into mixed-integer linear programming (MILP) solvers, thereby facilitating communication between CPU solvers and GPU-based ML models. They highlight the potential of leveraging solver statistics to adapt optimization approaches automatically, presenting a new area of research at the intersection between machine learning and mathematical optimization.

The authors show the different between approaches that work on specific problems, and those that are built with the aim of generalizing across heterogeneous instances, noting the trade-offs involved in this choice.




## 7. Conclusion and Future Directions

We have examined architectural paradigms, learning strategies, and empirical performance across three domains: general CO problems (with emphasis on TSP), MILP solvers, and SAT solvers.

This paper serve as a comprehensive comparative analysis of machine learning approaches for combinatorial optimization problems. We have introduces   architectural paradigms, learning strategies, and empirical performance across all domains: CO problems, SAT, MIPL solvers 

Several key insights emerge from our analysis:

1. **Graph Neural Networks** have emerged as a dominant architectural paradigm across all domains, thanks to their ability to process graph-structured data and capture the relational nature of CO problems.
2. **Hybrid neural-algorithmic approaches** that combine ML components with traditional algorithmic frameworks show the most promise in terms of balancing solution quality and computational efficiency.
3. **Learning strategies** vary across domains, with supervised learning via imitation being more common in MILP applications and reinforcement learning being more prevalent in general CO problems and SAT solvers.
4. **Generalization** across heterogeneous problem instances remains a significant challenge, with approaches like tree state parameterization in MILP showing promise in addressing this issue.
5. **Scalability** to larger problem instances is another common challenge, with computational overhead of neural networks potentially offsetting the benefits of improved decision-making.

This paper shows that this route can be developed in several way we can sight some of them:

1. **Developing more efficient neural architectures**
2. **Exploring hybrid learning strategies** 
3. **Investigating transfer learning and meta-learning** 
4. **Addressing the integration challenges** 
5. **Developing standardized benchmarks and evaluation methodologies** 

Machine learning for combinatorial optimization is a rapidly evolving field with significant potential for impact across various domains. While substantial progress has been made, there remain ample opportunities for further research and innovation.
