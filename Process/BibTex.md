@article{chung2025neural,
  title={Neural Combinatorial Optimization with Reinforcement Learning in Industrial Engineering: A Survey},
  author={Chung, K. T. and Lee, C. K. M. and Tsang, Y. P.},
  journal={Artificial Intelligence Review},
  volume={58},
  pages={130},
  year={2025},
  month={mar},
  publisher={Springer},
  doi={10.1007/s10462-024-11045-1},
  keywords={Neural combinatorial optimization; Reinforcement learning; Industrial engineering; Survey; Industry 5.0},
  annote={This comprehensive survey examines how reinforcement learning (RL) techniques are applied to neural combinatorial optimization (NCO) in industrial engineering contexts. The authors categorize existing approaches and highlight how NCO-RL algorithms can effectively solve hard combinatorial optimization problems while addressing the complex requirements of modern industry, including sustainability, adaptability, and human-centric design. The survey demonstrates how these methods are particularly valuable for real-time decision-making in Industry 5.0 applications.}
}

@article{mazyavkina2021reinforcement,
  title={Reinforcement Learning for Combinatorial Optimization: A Survey},
  author={Mazyavkina, Nina and Sviridov, Sergey and Ivanov, Sergei and Burnaev, Evgeny},
  journal={Computers \& Operations Research},
  volume={134},
  pages={105400},
  year={2021},
  month={may},
  publisher={Elsevier},
  doi={10.1016/j.cor.2021.105400},
  keywords={Reinforcement learning; Combinatorial optimization; Value-based methods; Policy-based methods; Neural networks},
  annote={This survey investigates the intersection of reinforcement learning and combinatorial optimization, highlighting how this approach is revolutionizing solutions for traditionally hard computational problems. The authors provide a comprehensive taxonomy of RL methods for combinatorial optimization, including value-based and policy-based algorithms, and discuss neural network architectures such as LSTMs, GRUs, and GNNs. A key insight is that these methods can significantly reduce computation time compared to classical heuristic approaches, making them increasingly relevant for real-world applications.}
}

@article{wang2024solving,
  title={Solving Combinatorial Optimization Problems with Deep Neural Network: A Survey},
  author={Wang, Feng and He, Qi and Li, Shicheng},
  journal={Tsinghua Science and Technology},
  volume={29},
  number={5},
  pages={1266--1282},
  year={2024},
  month={oct},
  publisher={IEEE},
  doi={10.26599/TST.2023.9010076},
  keywords={Deep neural networks; Combinatorial optimization; Constructive algorithms; Improvement algorithms; Pointer networks; Graph neural networks},
  annote={This survey categorizes deep neural network (DNN) approaches for combinatorial optimization problems into constructive and improvement algorithms. The authors highlight that despite significant advancements, there remains substantial room for improvement in neural network architectures due to limitations in feature extraction capability. A key advantage of these methods is their independence from expert knowledge, which has contributed to their growing popularity over the past decade. The paper provides a comprehensive overview of pointer network approaches, GNN-based methods, and algorithms that combine DNNs with traditional methods.}
}

@article{alvarez2017machine,
  title={A Machine Learning-Based Approximation of Strong Branching},
  author={Alvarez, Alejandro Marcos and Louveaux, Quentin and Wehenkel, Louis},
  journal={INFORMS Journal on Computing},
  volume={29},
  number={1},
  pages={185--195},
  year={2017},
  month={jan},
  publisher={INFORMS},
  doi={10.1287/ijoc.2016.0723},
  keywords={Mixed-integer programming; Machine learning; Branching rules; Supervised learning; ExtraTrees},
  annote={This paper presents a machine learning solution to improve branching decisions in Branch and Bound algorithms for Mixed Integer Linear Programming (MILP). Using an ExtraTrees model with carefully engineered features to represent the problem state, the authors demonstrate significant performance improvements, including a 3x speedup in some cases. The approach effectively approximates strong branching, which is computationally expensive but produces high-quality branching decisions, using a more efficient ML-based method that captures the essential aspects of the branching problem.}
}

@inproceedings{zarpellon2021parameterizing,
  title={Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies},
  author={Zarpellon, Giulia and Jo, Jason and Lodi, Andrea and Bengio, Yoshua},
  booktitle={Proceedings of the 35th AAAI Conference on Artificial Intelligence},
  pages={3931--3939},
  year={2021},
  month={feb},
  organization={AAAI Press},
  doi={10.1609/aaai.v35i5.16512},
  keywords={Branch and bound; Mixed integer programming; Branching policies; Deep learning; Generalization},
  annote={This paper introduces a novel approach to branch variable selection (BVS) in MILP solvers using deep neural networks. The authors parameterize B\&B search trees to capture the state of the optimization process and use imitation learning to train policies that generalize across heterogeneous problems. Their key insight is that representing the search tree state can help the model adapt branching decisions to different phases of the optimization process. Experimental results show that their TreeGate policy significantly outperforms baseline methods and reduces tree size by 27\% compared to traditional approaches without tree state representation.}
}

@article{scavuzzo2023machine,
  title={Machine Learning Augmented Branch and Bound for Mixed Integer Linear Programming},
  author={Scavuzzo, Lara and Aardal, Karen and Lodi, Andrea and Yorke-Smith, Neil},
  journal={European Journal of Operational Research},
  volume={305},
  number={3},
  pages={1089--1107},
  year={2023},
  month={apr},
  publisher={Elsevier},
  doi={10.1016/j.ejor.2022.10.039},
  keywords={Machine learning; Branch and bound; Mixed integer programming; Solver optimization; Adaptive learning},
  annote={This survey examines the integration of machine learning techniques into Mixed Integer Linear Programming (MILP) solvers, highlighting trends in instance representation, learning algorithms, and benchmarking practices. The authors address key research questions about handling heterogeneous instances versus specialized structures and identify challenges in CPU/GPU interaction for ML integration into MILP solvers. A significant contribution is their discussion of how solver statistics can be leveraged to adaptively select optimization approaches, presenting promising research directions at the intersection of machine learning and mathematical optimization.}
}

@article{guo2022machine,
  title={Machine Learning Methods in Solving the Boolean Satisfiability Problem},
  author={Guo, Wenxuan and Yan, Junchi and Zhen, Hui-Ling and Li, Xijun and Yuan, Mingxuan and Jin, Yaohui},
  journal={International Journal of Artificial Intelligence Tools},
  volume={31},
  number={2},
  pages={2203.04755},
  year={2022},
  month={mar},
  publisher={World Scientific},
  doi={10.48550/arXiv.2203.04755},
  keywords={Boolean satisfiability; Machine learning; Graph neural networks; End-to-end solvers; Supervised learning},
  annote={This paper examines the evolution of ML-based SAT solvers, from naive classifiers with handcrafted features to sophisticated end-to-end models. The authors show how graph neural networks (GNNs) can effectively encode SAT instances as literal-clause graphs to achieve high accuracy using supervised learning approaches. The paper also discusses limitations and potential improvements using unsupervised learning or reinforcement learning methods, and addresses the trustworthiness of these novel approaches compared to traditional solvers. It provides valuable insights into how deep learning architectures can improve SAT problem solving.}
}

@inproceedings{kurin2020can,
  title={Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver?},
  author={Kurin, Vitaly and Godil, Saad and Whiteson, Shimon and Catanzaro, Bryan},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9608--9620},
  year={2020},
  month={dec},
  organization={NeurIPS},
  doi={10.48550/arXiv.1909.11830},
  keywords={Reinforcement learning; SAT solving; Graph neural networks; Q-learning; Branching heuristics},
  annote={This paper investigates combining graph neural networks (GNNs) with Q-learning to develop generalizable branching heuristics for SAT solvers. Using a simple state representation and binary reward structure, the authors create Graph-Q-SAT, which reduces the number of iterations required to solve SAT problems by 2-3x. The approach demonstrates impressive generalization capabilities, performing well on problems up to 5x larger than training instances and transferring from satisfiable to unsatisfiable problems. Despite some limitations in wall-clock time reduction, the work provides compelling evidence that reinforcement learning with GNNs can effectively learn branching heuristics for combinatorial optimization.}
}

@article{khalil2022mip,
  title={MIP-GNN: A Data-Driven Framework for Guiding Combinatorial Solvers},
  author={Khalil, Elias B. and Morris, Christopher and Lodi, Andrea},
  journal={INFORMS Journal on Computing},
  volume={34},
  number={3},
  pages={1669--1684},
  year={2022},
  month={may},
  publisher={INFORMS},
  doi={10.1287/ijoc.2021.1120},
  keywords={Combinatorial optimization; Graph neural networks; Mixed integer programming; Bipartite graphs; Variable biases},
  annote={This paper introduces MIP-GNN, a framework that enhances combinatorial solvers using graph neural networks. The authors model mixed-integer linear programs (MILPs) as bipartite graphs and train GNNs to predict variable biases that guide solver decisions, such as node selection and warm-starting. This approach replaces traditional heuristics with learned policies that can generalize across problem instances. Experimental results show significant performance improvements in solving binary MILPs compared to default solver settings, demonstrating how deep learning tools can effectively enhance traditional combinatorial optimization methods.}
}

@inproceedings{liu2023combinatorial,
  title={Combinatorial Optimization with Automated Graph Neural Networks},
  author={Liu, Yang and Zhang, Peng and Gao, Yang and Zhou, Chuan and Li, Zhao and Chen, Hongyang},
  booktitle={Proceedings of the 37th AAAI Conference on Artificial Intelligence},
  pages={9224--9232},
  year={2023},
  month={feb},
  organization={AAAI Press},
  doi={10.1609/aaai.v37i8.26090},
  keywords={Neural architecture search; Graph neural networks; Combinatorial optimization; Message passing; Simulated annealing},
  annote={This work introduces AutoGNP, an automated framework for designing graph neural networks tailored to combinatorial optimization problems. Focusing on mixed integer linear programming (MILP) and quadratic unconstrained binary optimization (QUBO), AutoGNP leverages neural architecture search techniques to automatically optimize GNN structures without manual design. The authors introduce a novel two-hop message passing operator and employ simulated annealing for better convergence. The approach demonstrates superior performance over existing GNN-based combinatorial optimization methods while reducing the need for manual architecture tuning.}
}