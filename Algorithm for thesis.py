#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def plot_price_development(price_histories, window=50):
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(price_histories):
        moving_avg = np.convolve(history, np.ones(window)/window, mode='valid')
        plt.plot(range(len(moving_avg)), moving_avg, label=f'Firm {i+1}')
    plt.xlabel("Iterations")
    plt.ylabel("Price Moving Average")
    plt.title("Price Development Over Time (Moving Average)")
    plt.legend()
    plt.show()


# In[ ]:


class Firm_N:
    def __init__(self, mc, price_floor, price_cap, learning_rate=0.85, discount_factor=0.98):
        self.mc = mc
        self.price_floor = price_floor
        self.price_cap = price_cap
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}

    def get_state(self, prices):
        return tuple(int(p) for p in prices)  # Keep states as whole numbers

    def get_action(self, state, prices, firm_index):
        if state not in self.q_table:
            self.q_table[state] = {0: 0, 1: 0, 2: 0}  # Actions: 0 (decrease), 1 (same), 2 (increase)

        # Find the minimum price among competitors (exclude own price)
        competitor_prices = prices[:firm_index] + prices[firm_index+1:]  # Excludes current firm's price
        min_competitor_price = min(competitor_prices)

        # Check if the firm's price is at least 2 below the minimum competitor price
        if prices[firm_index] <= min_competitor_price - 2:
            # Apply exploration rate, but limit to actions 1 (same) or 2 (increase)
            if np.random.rand() < 0.5:  # Exploration rate
                return np.random.choice([1, 2])
            else:
                return max({k: v for k, v in self.q_table[state].items() if k in [1, 2]}, key=self.q_table[state].get)

        # Regular exploration-exploitation logic
        if np.random.rand() < 0.5:  # Exploration rate
            return np.random.choice([0, 1, 2])
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def get_learned_policy(self):
        policy = {}
        for state, actions in self.q_table.items():
            best_action = max(actions, key=actions.get)
            policy[state] = best_action
        return policy

    def update_q_value(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = {0: 0, 1: 0, 2: 0}
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)

        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action]
        )

    def calculate_profit(self, price, min_price_n):
        return (100 - price) * (price - 10) / min_price_n


def simulate_bertrand_rl(n_players=2, max_iterations=500000, price_floor=10, price_cap=55, start_prices=10):
    firms = [Firm_N(mc=10, price_floor=price_floor, price_cap=price_cap) for _ in range(n_players)]

    if isinstance(start_prices, (int, float)):
        prices = [start_prices] * n_players
    elif isinstance(start_prices, list):
        if len(start_prices) != n_players:
            raise ValueError("Length of start_prices must match n_players")
        prices = start_prices
    else:
        raise ValueError("start_prices must be an int, float, or list of length n_players")

    price_histories = [[p] for p in prices]
    rewards = [[] for _ in range(n_players)]
    Final_policy =[]

    for _ in range(max_iterations):
        states = [firm.get_state(prices) for firm in firms]
        actions = [firm.get_action(state, prices, j) for j, (firm, state) in enumerate(zip(firms, states))]

        # Determine new prices
        new_prices = []
        for j, (firm, action) in enumerate(zip(firms, actions)):
            if action == 0:  # Decrease price
                new_price = max(prices[j] - 1, price_floor)
            elif action == 2:  # Increase price
                new_price = min(prices[j] + 1, price_cap)
            else:
                new_price = prices[j]
            new_prices.append(new_price)

        # Find the minimum price and the number of firms setting it
        min_price = min(new_prices)
        min_price_n = new_prices.count(min_price)

        # Compute profit ONCE (only firms with min_price get this)
        profit_value = (100 - min_price) * (min_price - 10) / min_price_n

        # Assign profits (only firms setting min_price receive profit_value)
        profits = [profit_value if new_prices[j] == min_price else 0 for j in range(n_players)]

        # Update Q-values
        for j, firm in enumerate(firms):
            next_state = firm.get_state(new_prices)
            firm.update_q_value(states[j], actions[j], profits[j], next_state)

        # Store results
        for j in range(n_players):
            price_histories[j].append(new_prices[j])
            rewards[j].append(profits[j])

        prices = new_prices
        

    for firm in firms:
        Final_policy.append(firm.get_learned_policy())

    return price_histories, Final_policy


# In[ ]:


n = input("Define the number of players")
iterations = input("How many iterations should the algorithm conduct")
prices, Final_Policy = simulate_bertrand_rl(n_players=n, max_iterations= iterations, price_floor=10, price_cap=55, start_prices=10)
plot_price_development(prices, window = 1000)

