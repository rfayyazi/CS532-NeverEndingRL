import numpy as np


class AppleIsland:
    def __init__(self, D, growth_times):
        self.D = D
        self.grid = [Square() for _ in range(D)]

        # plant trees
        # self.tree_indices = np.random.choice(range(D), len(growth_times), replace=False)
        self.tree_indices = [0, int(D/2), D-1]
        for i, j in enumerate(self.tree_indices):
            self.grid[j].apple = True
            self.grid[j].t = growth_times[i]
            self.grid[j].growth_time = growth_times[i]

    def get_states(self, agents):
        grid_counts = [0 for _ in range(self.D)]
        for agent in agents:
            grid_counts[agent.position] += 1
        states = [grid_counts[:] for _ in range(len(agents))]
        for i, s in enumerate(states):
            s.append(agents[i].position)
        states = [np.array(s) for s in states]
        return states

    def transition(self, agents, actions):
        # p(s', r|s, a)
        rewards = np.zeros(len(agents))
        tree_pickers = {}  # collect agents that want to pick from each tree
        for i, agent in enumerate(agents):
            if actions[i] == 2:  # agent wants to pick apple
                if self.grid[agent.position].apple:  # if the agent is on a square with an apple
                    if agent.position in tree_pickers.keys():
                        tree_pickers[agent.position].append(i)
                    else:
                        tree_pickers[agent.position] = [i]
            else:
                agent.position = self.move_agent(agent.position, actions[i])

        for i, square in enumerate(self.grid):
            if i in tree_pickers.keys():
                successful_agent = np.random.choice(tree_pickers[i])
                rewards[successful_agent] = 1.0
                square.apple = False
                square.t = 0
            else:
                square.t += 1
                if square.t >= square.growth_time:
                    square.apple = True

        return rewards

    def move_agent(self, position, action):
        if action == 0 and position != 0:  # legal move left
            return position - 1
        elif action == 1 and position != (self.D - 1):  # legal move right
            return position + 1
        else:  # illegal move
            return position


class Square:
    def __init__(self):
        self.apple = False
        self.t = 0
        self.growth_time = np.inf


if __name__ == "__main__":
    p=0