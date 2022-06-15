import copy


class Board:
    def __init__(self, reward_func=None, grid=None, invalid_act_reward=-10):
        if grid is None:
            self._grid = {'00': 0, '01': 0, '02': 0, '03': 0,
                          '10': 0, '11': 0, '12': 0, '13': 0,
                          '20': 0, '21': 0, '22': 0, '23': 0,
                          '30': 0, '31': 0, '32': 0, '33': 0}
            self._random_spawn()
        else:
            self._grid = grid
        self._prev_grid = self._grid
        self._reward_func = reward_func if reward_func else lambda *args: self._current_reward
        self._invalid_act_reward = invalid_act_reward
        self._state_changed = False
        self._current_reward = 0
        self._total_reward = 0
        self._action_map = {1: self._push_up, 0: self._push_down,
                            3: self._push_left, 2: self._push_right}

    def to_np(self):
        import numpy as np
        return np.array([[self._grid['00'], self._grid['01'], self._grid['02'], self._grid['03']],
                         [self._grid['10'], self._grid['11'], self._grid['12'], self._grid['13']],
                         [self._grid['20'], self._grid['21'], self._grid['22'], self._grid['23']],
                         [self._grid['30'], self._grid['31'], self._grid['32'], self._grid['33']]], dtype=np.int32)

    def _random_spawn(self, prob_of_4=0.1):
        import numpy as np
        zeros = [el for el in self._grid if not self._grid[el]]
        position = np.random.choice(zeros)
        is_4 = np.random.uniform(0, 1)
        new_spawn = 4 if is_4 < prob_of_4 else 2
        self._grid[position] = new_spawn

    def _push(self, l):
        c = 0  # current index
        has_merged = False
        while c < 4:
            if self._grid[l[c]] == 0:
                nnz = c  # nearest nonzero element
                while nnz < 4 and self._grid[l[nnz]] == 0:
                    nnz += 1  # increment while there are 0's
                if nnz == 4:
                    break  # if nnz is 4 then the entire list after c has 0's => task is finished
                self._grid[l[c]] = self._grid[l[nnz]]
                self._grid[l[nnz]] = 0
                if c > 0 and not has_merged and self._grid[l[c - 1]] == self._grid[l[c]]:
                    self._grid[l[c - 1]] += self._grid[l[c - 1]]
                    self._current_reward += self._grid[l[c - 1]]
                    self._grid[l[c]] = 0
                self._state_changed = True
                has_merged = False
            if c < 3 and self._grid[l[c]] == self._grid[l[c + 1]]:
                has_merged = True
                self._grid[l[c]] += self._grid[l[c]]
                self._current_reward += self._grid[l[c]]
                self._grid[l[c + 1]] = 0
                self._state_changed = True
            c += 1
        self._total_reward += self._current_reward

    def _push_all(self, lists):
        self._current_reward = 0
        for l in lists:
            self._push(l)

    def _push_up(self):
        self._push_all([['00', '10', '20', '30'], ['01', '11', '21', '31'], ['02', '12', '22', '32'], ['03', '13', '23', '33']])

    def _push_down(self):
        self._push_all([['30', '20', '10', '00'], ['31', '21', '11', '01'], ['32', '22', '12', '02'], ['33', '23', '13', '03']])

    def _push_left(self):
        self._push_all([['00', '01', '02', '03'], ['10', '11', '12', '13'], ['20', '21', '22', '23'], ['30', '31', '32', '33']])

    def _push_right(self):
        self._push_all([['03', '02', '01', '00'], ['13', '12', '11', '10'], ['23', '22', '21', '20'], ['33', '32', '31', '30']])

    def print_grid(self):
        print(f"{self._grid['00']} {self._grid['01']} {self._grid['02']} {self._grid['03']}")
        print(f"{self._grid['10']} {self._grid['11']} {self._grid['12']} {self._grid['13']}")
        print(f"{self._grid['20']} {self._grid['21']} {self._grid['22']} {self._grid['23']}")
        print(f"{self._grid['30']} {self._grid['31']} {self._grid['32']} {self._grid['33']}")

    def _can_push(self):
        return self._cp('00', '01') or self._cp('01', '02') or self._cp('02', '03') or \
               self._cp('10', '11') or self._cp('11', '12') or self._cp('12', '13') or \
               self._cp('20', '21') or self._cp('21', '22') or self._cp('22', '23') or \
               self._cp('30', '31') or self._cp('31', '32') or self._cp('32', '33') or \
               self._cp('00', '10') or self._cp('01', '11') or self._cp('02', '12') or self._cp('03', '13') or \
               self._cp('10', '20') or self._cp('11', '21') or self._cp('12', '22') or self._cp('13', '23') or \
               self._cp('20', '30') or self._cp('21', '31') or self._cp('22', '32') or self._cp('23', '33')

    def _cp(self, c1, c2):
        return self._grid[c1] == 0 or self._grid[c2] == 0 or self._grid[c1] == self._grid[c2]

    def step(self, action):
        if action not in self._action_map:
            raise ValueError(f'action {action} not in map')
        self._state_changed = False
        self._action_map.get(action)()  # performs the action
        if self._state_changed:
            self._random_spawn()
            reward = self._reward_func(self._prev_grid, self._grid)
        else:
            self._current_reward = self._invalid_act_reward
            reward = self._invalid_act_reward
        self._prev_grid = copy.deepcopy(self._grid)
        if not self._can_push():
            return self.to_np(), self._total_reward, True
        else:
            return self.to_np(), reward, False

    def current_reward(self):
        return self._current_reward

    def total_reward(self):
        return self._total_reward

class Game:
    def __init__(self, action_func, reward_func=None, show=True, grid=None, delay=None):
        import copy
        self.board = Board(grid)
        self.action_map = {0: self.board._push_up, 1: self.board._push_down,
                           2: self.board._push_left, 3: self.board._push_right}
        self.action_func = action_func
        self.reward_func = reward_func if reward_func else self.default_game_reward
        self.show = show
        self.delay = delay
        self.prev_grid = copy.deepcopy(self.board._grid)

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
        return self.board._current_reward

    def step(self):
        import os
        import copy
        if self.show:
            os.system('clear')
        action = self.action_func()
        self.board._state_changed = False
        self.action_map.get(action)()  # performs the action
        if self.action_func == Game.random_input:
            time.sleep(self.delay)
        if self.board._state_changed:
            self.board._random_spawn()
        if self.show:
            self.board.print_grid()
        reward = self.reward_func(self.prev_grid, self.board._grid)
        self.prev_grid = copy.deepcopy(self.board._grid)
        return action, self.prev_grid, reward, not self.board._can_push()


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
