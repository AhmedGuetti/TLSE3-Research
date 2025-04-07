# A Machine Learning-Based Approximation of Strong Branching
#Alejandro_Marcos_Alvarez, #Quentin_Louveaux, #Louis_Wehenkel

This paper presents a new approach at the time to variable branching in B&B to solve MILP. This approach imitate the decision taken by a good branching strategy namely strong branching and reliable branching, with a faster approximation. This approximation function is created by a ML technique from a set of B&B made by strong branching, using heuristic methods. The experiments preformed on randomly generated MIPLIB problems benchmark data set, and it show a promising result.
# The Innovation

The authors propose a machine learning approach to create a "learned" branching strategy that imitates strong branching decisions but executes much faster, in this paper they propose a two-phase approach: 
1. **Training:** where we train our model on random MILP problems recording each decision has been made using a heuristic solutions and minimizing our learning function based on the feature chased to represent the subproblem
2. **Application phase**: we use the learned heuristic to solve new problem.

in this paper the authors proposed using *ExtraTrees* algorithm that is an optimal algorithm based on random forest that is highly depend on the $k$ initial parameter or learning rate. We train this algorithm on $10^5$ samples from a dataset of $7×10^7$ strong branching decisions

# Features
In this paper the feature are the most important aspect of the solution where they should be efficient and they have to describe precisely the subproblem we would encounter.
## Properties
They identify three essential properties these features should have:
1. **Size Independence**: Features must be independent of the problem instance size, allowing a single learned branching strategy to work across different problem sizes.
2. **Invariance to Irrelevant Changes**: Features should be invariant to changes like row/column permutations.
3. **Scale Independence**: Features should remain identical if problem parameters are multiplied by some factor. 
## Categories
The features (denoted as $\phi_i$) are divided into three categories:
1. **Static Problem Features:** These are computed once from parameters c, A, and b, representing the static state of the problem
2. **Dynamic Problem Features**: These relate to the solution at the current B&B (Branch and Bound) node
3. **Dynamic Optimization Features**: These represent the overall state of the optimization process

some measurement are shown here 
1. **Cost Function Features:** 
	They contain the sign of $c_i$ , and the Normalized magnitude 
	$$\frac{|c_i|}{\sum_{j : c_j \geq 0} |c_j|} \text{ and } \frac{|c_i|}{\sum_{j : c_j < 0} |c_j|}$$
2. **Constraint Matrix Features:**
	we have three measures to to describe a variable $i$'s relationship with the constraint matrix $A$
	- $M_i^1$:  Measures how much a variable contributes to constraint violations
	- $M_i^2$:  Models the relationship between cost coefficient $c_i$ and constraint coefficients
	- $M_i^3$: Represents intervariable relationships within constraints
3. **Objective Increase Statistics**:
	When branching is performed on a variable, the algorithm stores the resulting objective increases
4. **Branching History**:
	The final feature in this category is the number of times variable $i$ has been chosen as a branching variable, normalized by the total number of branching performed

# Results and Performance

This hybrid method showed a promising result where it have a fine ~91% gap from the Strong branching algorithm while having a more than 80% faster result. witch is not a bad trade off. Moreover this paper shows that this new method can explore more node than the strong brunching if no time limitation is set, after the test of using the CPLEX's cuts we outperform all the other method by stunning x3 more speed, all the result are tested on the MILPLIB benchmark and compared against :
- **random branching**
- **most-infeasible**
- **nonchimerical**
- **full strong branching**
- **reliability branching**
# Limitations

The main problem with this method is that it's more optimized with problems that are similar to the training data witch lead performance varies across problem types so for that it run poorly on problem that are not well structured in the training data, witch lead to a conclusion that this method  perform better when the feature are well, representing the problem in hand, in the other the engineering of the feature should be done in a very careful way to represent the main aspect of the problem in hand. 