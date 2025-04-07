# TSP and RL 
## Neural combinatorial optimization with reinforcement learning in industrial engineering: a survey
#Chung #Lee #Tsang
combinatorial optimization (CO) problems are develops to support the wide complicated decision-making to increase cost-efficiency and  productivity, solving the CO problem leveraging the power of Machine learning is a growing trend to solve the uncertainty of heuristic method and the over-complexity of exact algorithm we call them neural combinatorial optimization (NCO), this paper explore the use of Reinforcement learning to solve CO in an industrial engineering contexts, , they show significant promise for real-time decision-making particularly for the transition to Industry 5.0 which emphasizes sustainability, adaptability, and human factors. 

Keyword:
#NCO #NCO-RL #TSP #routing-problem #production-scheduling #inventory-control #B-and-B #MILP #MDP #MCTS #Bellman-Equation #DQN
## Reinforcement learning for combinatorial optimization: A survey
#Mazyavkina #Sviridov  #Ivanov  #Burnaev

The survey investigate the intersection of reinforcement learning (RL) and combinatorial optimization (CO), showing how this approach is revolutionizing solutions to traditionally hard computational problems. reducing  computation time compared to classical heuristic methods, making them increasingly relevant for real-world applications.

keyword:
#MILP #TSP #Max-cut #BPP #MVC #MIS #MDP #Q-value #policy #LSTM #GRU #GNN #RNN #PN 

## Solving Combinatorial Optimization Problems with Deep Neural Network: A Survey
#Wang #He #Li 
This survey shows that despite the advancement on the use of DNN in solving COP we still have a long margin of improvement on the mainly on the architecture of the Neural Network due to the limitation of feature extraction, the fact that those new methods are not relying on the expert knowledge made them popular in the last decade.  

#PN #GNN #DNN #TSP #Q-value #DNQ 

# B&B hybrid methods
## A Machine Learning-Based Approximation of Strong Branching
#Alejandro_Marcos_Alvarez, #Quentin_Louveaux, #Louis_Wehenkel
This paper show a ML solution to solve decide the branching on a Branch and bound solution mainly the MILP. using an ExtraTree as a model and carefully engineered feature to represent the problem in hand shoeing a stunning 3 time boost in time in some cases, and over all good performance.

## Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies
#Giulia_Zarpellon, #Jason_Jo, #Andrea_Lodi, #Yoshua_Bengio7 

This paper tries to show an implementation using ML to impose branch variable selection (BVS) in MILP solvers therefore enhancing the branch and bound algorithm, where a single poor decision can lead in  significantly increase the search tree size (nodes), in this paper they introduces a DNN architecture to learn those decision using Imitation Learning, encoding the data in a $Tree_t$  parameterization, they also compared this model to the GCNN model. 

## Machine Learning Augmented Branch and Bound for Mixed Integer Linear Programming
#Lara_Scavuzzo , #Karen_Aardal , #Andrea_Lodi , #Neil_Yorke-Smith

This surveys paper integration of ML into MLP solvers, highlight the emerging field with big potential. The authors discussed the different trends in instance representation, learning algorithms and benchmarking. They address key research questions about  heterogeneous instances versus specialized structures. it also identify the CPU/GPU interaction in the challenge of ML integration into MILP. They also shows promising potential that leverage solver statistics to adapt optimization approaches automatically,  presenting new research opportunities in this intersection of ML and mathematical optimization.

# SAT Solvers
## Machine Learning Methods in Solving the Boolean Satisfiability Problem
#Wenxuan_Guo #Junchi_Yan #Hui-Ling_Zhen #Xijun_Li #Mingxuan_Yuan #Yaohui_Jin
This paper examine the evolution of the ML-SAT (Machine learning - Boolean Satisfiability Problem)solver from naive classifier with handcrafted feature developed by human expert to end-to-end solver, the authors shows how it is possible to solve SAT problem with accuracy using a supervised learning on a GNN, by encoding the SAT instance into a literal-clause graph (LIG), discussing the limitation and the important for other Solver using unsupervised learning or RL  approach, moreover the trust worthy of this approach.

## Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver?
#Vitaly_Kurin #Saad_Godil #Shimon_Whiteson #Bryan_Catanzaro

This paper investigate the ability to leverage the power of GNN and reinforcement learning to make Generalized branching heuristic, the authors showed state representation and the graph building, moreover they explores some coming DQN method to enhance the result. This work used simple state representation and require elaborate reward shaping, the method presented contain a lot of weakness that make heuristic method more adaptable for the industries, manly the wall-clock reduction. The authors where able to show a 2-3X reduction in iterations for problems up to 5X larger and 1.5-2X from SAT to unSAT.

