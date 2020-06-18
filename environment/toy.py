import logging
import random

import torch

#
logger = logging.getLogger('experiment')


class Toy:
    def __init__(self, less_likely=[]):
        #
        self.state = torch.zeros(1, 12)
        self.logger = logging.getLogger('experiment')
        self.state[0, 0] = 1
        self.state[0, 10] = 1
        self.color_cor_direction = 1
        self.y = 0
        self.color_cor_1 = 0.90
        self.color_cor_2 = 0.80
        self.color_cor = self.color_cor_1
        self.less_likely = less_likely
        self.logger.info("Cor values = %f, %f", self.color_cor_1, self.color_cor_2)

    def take_action(self, action):

        assert (action == -1 or action == 1)
        #
        self.state = self.state.clone()

        current_non_zero = torch.argmax(self.state[0, 0:10])

        rand_var = random.random()
        self.state[0, current_non_zero] = 0
        if rand_var > 0.65 and rand_var < 0.80:
            current_non_zero = (current_non_zero + 1 * action) % 10

        if rand_var > 0.80 and rand_var < 0.90:
            current_non_zero = (current_non_zero + 2 * action) % 10

        if rand_var > 0.90 and rand_var < 0.95:
            current_non_zero = (current_non_zero + 3 * action) % 10

        if rand_var > 0.95 and rand_var < 0.98:
            current_non_zero = (current_non_zero + 4 * action) % 10

        if rand_var > 0.98:
            current_non_zero = (current_non_zero + 5 * action) % 10

        assert (int(current_non_zero) in list(range(10)))

        if current_non_zero in self.less_likely:
            rand_flip_less_likely = random.random()
            if rand_flip_less_likely > 0.2:
                current_non_zero = (current_non_zero + 1) % 10

        self.state[0, current_non_zero] = 1

        color_index = torch.argmax(self.state[0, 10:12]) + 10
        self.state[0, color_index] = 0

        if current_non_zero < 5:
            self.y = 0
        else:
            self.y = 1

        # 
        rand_flip = random.random()
        if rand_flip > 0.75:
            self.y = (self.y + 1) % 2

        color_flip = random.random()

        hidden_color = random.random()
        # if hidden_color > 0.90:
        if color_flip > self.color_cor:
            self.state[0, 10 + ((self.y + 1) % 2)] = 1
        else:
            self.state[0, 10 + self.y] = 1

        increment = random.random()
        if increment > 0.9999:
            if self.color_cor == self.color_cor_1:
                self.color_cor = self.color_cor_2
            elif self.color_cor == self.color_cor_2:
                self.color_cor = self.color_cor_1
            else:
                assert (False)

    def get_target(self):

        return torch.tensor(self.y).view(1, 1)

    def get_state(self):

        return self.state

    def force_change(self):
        if self.color_cor == self.color_cor_1:
            self.color_cor = self.color_cor_2
        elif self.color_cor == self.color_cor_2:
            self.color_cor = self.color_cor_1

    #
    def set_random_state(self):

        self.state = self.state.clone()

        current_non_zero = torch.argmax(self.state[0, 0:10])

        self.state[0, current_non_zero] = 0

        current_non_zero = random.randint(0, 9)
        assert (int(current_non_zero) in list(range(10)))

        if current_non_zero in self.less_likely:
            rand_flip_less_likely = random.random()
            if rand_flip_less_likely > 0.2:
                # print("gets here")
                current_non_zero = (current_non_zero + 1) % 10

        self.state[0, current_non_zero] = 1

        color_index = torch.argmax(self.state[0, 10:12]) + 10
        self.state[0, color_index] = 0

        if current_non_zero < 5:
            self.y = 0
        else:
            self.y = 1

        rand_flip = random.random()
        if rand_flip > 0.75:
            self.y = (self.y + 1) % 2

        color_flip = random.random()

        if color_flip > self.color_cor:
            self.state[0, 10 + ((self.y + 1) % 2)] = 1
        else:
            self.state[0, 10 + self.y] = 1
