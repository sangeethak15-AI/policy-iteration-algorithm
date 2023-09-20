# EXPERIMENT 3: POLICY ITERATION ALGORITHM

## AIM
To implement a policy iteration algorithm for the given MDP

## PROBLEM STATEMENT
The problem statement defines a Stochastic Bandit walk environment with five states excluding the Goal state and the hole state.
### State Space:
{0(HOLE),1,2,3,4,5,6(GOAL)} <br>
Thus it includes 2 terminal states(0 and 6) and 5 non-terminal states.

### Action Space:
Two actions 0 and 1 are possible, <br>
{0(LEFT),1(RIGHT)}

### Reward Function:
* Reaches Goal state: +1
* Otherwise: 0
### Tranisition Probability:
* 50% - Agent moves in the desired direction
* 33.33% - Agent stays in the same state
* 16.66% - Agent movies in orthogonal direction

## POLICY ITERATION ALGORITHM
1. Initialization: 
Start with an initial policy, which can be random or arbitrary.Initialize a value function (e.g., V(s)) for each state in the MDP.
2. Policy Evaluation:
Evaluate the current policy by estimating the value function V(s) for each state using iterative methods 
Update the value function until it converges or reaches a predefined threshold.
3. Policy Improvement:
Based on the updated value function, greedily improve the policy by selecting the action in each state that maximizes the expected cumulative reward.
4. Iteration:
Repeat the Policy Evaluation and Policy Improvement steps iteratively until the policy no longer changes significantly, indicating that an optimal policy has been found.
## POLICY IMPROVEMENT FUNCTION
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
    new_pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```

## POLICY ITERATION FUNCTION
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi=lambda s:{s:a for s,a in enumerate(random_actions)}[s]
    while True:
      old_pi={s:pi(s) for s in range(len(P))}
      V=policy_evaluation(pi,P,gamma,theta)
      pi=policy_improvement(V,P,gamma)
      if old_pi=={s:pi(s) for s in range(len(P))}:
        break

    return V, pi
```
## OUTPUT
### Optimal Policy
<img width="545" alt="image" src="https://github.com/Shavedha/policy-iteration-algorithm/assets/93427376/382b9dc9-53c1-4377-b199-3220b10f29c8">

### Optimal Value Function
<img width="586" alt="image" src="https://github.com/Shavedha/policy-iteration-algorithm/assets/93427376/9e751476-47be-49da-81cc-a4529d936df0">

### Success Probability
<img width="477" alt="image" src="https://github.com/Shavedha/policy-iteration-algorithm/assets/93427376/1c281d2c-8cf9-4aee-84e6-a0d6ad1b07b4">

## RESULT
Thus a program is developed to perform policy iteration.
