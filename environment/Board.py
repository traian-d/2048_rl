class Board:
    def __init__(self, grid=None):
        if grid is None:
            self.grid = {'00': 0, '01': 0, '02': 0, '03': 0,
                         '10': 0, '11': 0, '12': 0, '13': 0,
                         '20': 0, '21': 0, '22': 0, '23': 0,
                         '30': 0, '31': 0, '32': 0, '33': 0}
            self.random_spawn()
        else:
            self.grid = grid
        self.state_changed = False
        self.current_reward = 0

    def random_spawn(self, prob_of_4=0.1):
        import numpy as np
        zeros = [el for el in self.grid if not self.grid[el]]
        position = np.random.choice(zeros)
        is_4 = np.random.uniform(0, 1)
        new_spawn = 4 if is_4 < prob_of_4 else 2
        self.grid[position] = new_spawn

    def push(self, l):
        c = 0  # current index
        has_merged = False
        while c < 4:
            if self.grid[l[c]] == 0:
                nnz = c  # nearest nonzero element
                while nnz < 4 and self.grid[l[nnz]] == 0:
                    nnz += 1  # increment while there are 0's
                if nnz == 4:
                    break  # if nnz is 4 then the entire list after c has 0's => task is finished
                self.grid[l[c]] = self.grid[l[nnz]]
                self.grid[l[nnz]] = 0
                if c > 0 and not has_merged and self.grid[l[c - 1]] == self.grid[l[c]]:
                    self.grid[l[c - 1]] += self.grid[l[c - 1]]
                    self.current_reward += self.grid[l[c - 1]]
                    self.grid[l[c]] = 0
                self.state_changed = True
                has_merged = False
            if c < 3 and self.grid[l[c]] == self.grid[l[c + 1]]:
                has_merged = True
                self.grid[l[c]] += self.grid[l[c]]
                self.current_reward += self.grid[l[c]]
                self.grid[l[c + 1]] = 0
                self.state_changed = True
            c += 1

    def push_all(self, lists):
        self.current_reward = 0
        for l in lists:
            self.push(l)

    def push_up(self):
        self.push_all([['30', '20', '10', '00'], ['31', '21', '11', '01'], ['32', '22', '12', '02'], ['33', '23', '13', '03']])

    def push_down(self):
        self.push_all([['00', '10', '20', '30'], ['01', '11', '21', '31'], ['02', '12', '22', '32'], ['03', '13', '23', '33']])

    def push_left(self):
        self.push_all([['03', '02', '01', '00'], ['13', '12', '11', '10'], ['23', '22', '21', '20'], ['33', '32', '31', '30']])

    def push_right(self):
        self.push_all([['00', '01', '02', '03'], ['10', '11', '12', '13'], ['20', '21', '22', '23'], ['30', '31', '32', '33']])

    def print_grid(self):
        print(f"{self.grid['00']} {self.grid['01']} {self.grid['02']} {self.grid['03']}")
        print(f"{self.grid['10']} {self.grid['11']} {self.grid['12']} {self.grid['13']}")
        print(f"{self.grid['20']} {self.grid['21']} {self.grid['22']} {self.grid['23']}")
        print(f"{self.grid['30']} {self.grid['31']} {self.grid['32']} {self.grid['33']}")

    def can_push(self):
        return self.cp('00', '01') or self.cp('01', '02') or self.cp('02', '03') or \
               self.cp('10', '11') or self.cp('11', '12') or self.cp('12', '13') or \
               self.cp('20', '21') or self.cp('21', '22') or self.cp('22', '23') or \
               self.cp('30', '31') or self.cp('31', '32') or self.cp('32', '33') or \
               self.cp('00', '10') or self.cp('01', '11') or self.cp('02', '12') or self.cp('03', '13') or \
               self.cp('10', '20') or self.cp('11', '21') or self.cp('12', '22') or self.cp('13', '23') or \
               self.cp('20', '30') or self.cp('21', '31') or self.cp('22', '32') or self.cp('23', '33')

    def cp(self, c1, c2):
        return self.grid[c1] == 0 or self.grid[c2] == 0 or self.grid[c1] == self.grid[c2]


class Game:
    def __init__(self, action_func, reward_func=None, show=True, grid=None, delay=None):
        import copy
        self.board = Board(grid)
        self.action_map = {0: self.board.push_up, 1: self.board.push_down,
                           2: self.board.push_left, 3: self.board.push_right}
        self.action_func = action_func
        self.reward_func = reward_func if reward_func else self.default_game_reward
        self.show = show
        self.delay = delay
        self.prev_grid = copy.deepcopy(self.board.grid)

    @staticmethod
    def random_input():
        import numpy as np
        return np.random.randint(0, 4)

    @staticmethod
    def keyboard_input():
        k = input()
        return {'w': 1, 'a': 3, 's': 0, 'd': 2}.get(k)

    @staticmethod
    def zero_max_reward(prev_grid, current_grid):
        prev_zeros = [el for el in prev_grid if not prev_grid[el]]
        current_zeros = [el for el in current_grid if not current_grid[el]]
        return len(current_zeros) - len(prev_zeros)

    def default_game_reward(self, *args):
        return self.board.current_reward

    def step(self):
        import os
        import copy
        if self.show:
            os.system('clear')
        action = self.action_func()
        self.board.state_changed = False
        self.action_map.get(action)()  # performs the action
        if self.action_func == Game.random_input:
            time.sleep(self.delay)
        if self.board.state_changed:
            self.board.random_spawn()
        if self.show:
            self.board.print_grid()
        reward = self.reward_func(self.prev_grid, self.board.grid)
        self.prev_grid = copy.deepcopy(self.board.grid)
        return action, self.prev_grid, reward, not self.board.can_push()


if __name__ == '__main__':
    import time
    import numpy as np
    grid = {'00': 0, '01': 0, '02': 2, '03': 4,
            '10': 0, '11': 0, '12': 0, '13': 4,
            '20': 0, '21': 0, '22': 0, '23': 8,
            '30': 0, '31': 8, '32': 2, '33': 8}
    game = Game(action_func=Game.keyboard_input, show=True, grid=grid)
    action, prev_grid, reward, done = game.step()
    print(f'{reward} {done}')
    while not done:
        action, prev_grid, reward, done = game.step()
        print(f'{reward} {done}')


    # counters = []
    # states = []
    # start_time = time.time()
    # for i in range(5000):
    #     if not i % 1000:
    #         print(i)
    #     counter, final_state = game(delay=0.0, action_func=random_input, show=False)
    #     counters.append(counter)
    #     states.append(final_state)
    # print(time.time() - start_time)
    # with open('counts.csv', mode='w') as f:
    #     f.write('counts,cell,value\n')
    #     for i in range(len(counters)):
    #         for cell in states[i].keys():
    #             f.write('{},{},{}\n'.format(str(counters[i]), 'c' + cell, str(states[i][cell])))
