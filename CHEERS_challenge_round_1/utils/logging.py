"""
Utility classes and functions for logging data
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "08-03-2021"

import numpy as np

class Logger():
    pass

class TrainTracker():
"""Keep an overview over training and evaluation of a model"""

    self.loggers = []
    self.updates = []
    self.eval_updates[]

    def __init__(self, *loggers: list[Logger], epoch: int = 1, epoch_function = np.mean, eval_epoch_function = np.mean):

        self.epoch = epoch
        self.update = 0
        self.epoch_function = epoch_function
        self.eval_epoch_function = eval_epoch_function

        for idx, logger in enumerate(loggers):
            self.loggers.append((idx, logger))
        print_overview()

    def epoch_step(self):
        result = self.episode_function(self.updates)
        for logger in loggers:
            logger[1].epoch_step(result)

        self.updates = []

    def step(self, item):
        self.updates.append(item)
        for logger in loggers:
            logger[1].step(item)

    def eval_step(self, item):
        self.eval_updates.append(item)

    def eval_epoch_step(self):
        """Calculates the evaluation score for this epoch and updates
        """
        result = self.eval_epoch_function(self.updates)
        for logger in loggers:
            logger[1].eval_epoch_step(item)


    def __call__(self, item):
        """Calls "step" """
        self.step(item)

    def print_overview(self):
        """Print an overview of the Tracker"""

        print("Logger ID | Logger Name")
        for logger in self.loggers:
            print(f"    {logger[0]:02}    | {str(logger[1])[:50]}")