import numpy as np


class ExponentialDecayingMovingAverage:
    def __init__(self, alpha=0.5):  # TODO: make alpha an official hyperparameter
        self.alpha = alpha
        self.edma = None

    def update(self, data_point):
        if self.edma is None:
            self.edma = data_point
        else:
            self.edma = self.alpha * data_point + (1 - self.alpha) * self.edma
        return self.edma

    def get_edma(self):
        return self.edma


class StatsTracker:
    def __init__(self, task_names) -> None:
        self.task_names = task_names

        self.raw_stats = {}
        self.raw_stats["all"] = self.init_raw_stats(task_names)
        self.raw_stats["epoch"] = self.init_raw_stats(task_names)
        self.epoch_edma = self.init_raw_stats(task_names, edma=True)

        self.latest_epoch_agg = self.calc_latest_epoch_agg()  # should be zeros??

    def init_raw_stats(self, task_names, edma=False):
        raw_stats = {}
        for name in task_names:
            if edma:
                raw_stats[
                    name
                ] = ExponentialDecayingMovingAverage()  # TODO: set alpha properly
            else:
                raw_stats[name] = []
        return raw_stats

    def append_stat(self, task_name, stat):
        self.raw_stats["all"][task_name].append(stat)
        self.raw_stats["epoch"][task_name].append(stat)

    def reset_epoch_stats(self):
        # record current epoch stats
        self.latest_epoch_agg = self.calc_latest_epoch_agg()
        # update edma
        self.update_edma(self.latest_epoch_agg)
        # reset
        self.raw_stats["epoch"] = self.init_raw_stats(self.task_names)

    def get_agg_stats(self, agg_func):
        agg_stats = {}
        for time_key in ["all", "epoch"]:
            agg_stats[time_key] = {}
            for task_key in self.raw_stats[time_key].keys():
                if len(self.raw_stats[time_key][task_key]) < 1:
                    agg_stat = None
                else:
                    agg_stat = agg_func(self.raw_stats[time_key][task_key])
                agg_stats[time_key][task_key] = agg_stat
        return agg_stats

    def calc_latest_epoch_agg(self):
        agg_stats = self.get_agg_stats(lambda x: np.mean(x))
        latest_epoch_agg = agg_stats["epoch"]
        return latest_epoch_agg

    def update_edma(self, latest_epoch_agg):
        for task_key, task_stat in latest_epoch_agg.items():
            if task_stat is not None:
                self.epoch_edma[task_key].update(task_stat)

    def get_task_edma(self, task_name):
        return self.epoch_edma[task_name].get_edma()

    def get_task_epoch_success_rate(self, task_name):
        return self.latest_epoch_agg[task_name]

    def get_task_epoch_agg(self, task_name):
        return self.latest_epoch_agg[task_name]
