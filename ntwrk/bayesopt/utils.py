import math
import torch

def compute_reward(monitor_interval):
    """
    Get an Estimate of the throughput of the network
    using on the last few packets (the lag)
    """
    loss = len(monitor_interval.ack)/len(monitor_interval.sent)
    latency = sum(monitor_interval.rtts)/len(monitor_interval.rtts)

    ##################################
    ## Need to calculate throughput ##
    ##################################

    pass


class MonitorInterval(object):
    """docstring for MonitorInterval."""

    def __init__(self, cwnd):
        self.cwnd = cwnd
        self.sent = []
        self.ack = []
        self.rtts = []


class MonitorHistory(object):
    """docstring for MonitorHistory."""

    def __init__(self, history_length=10):
        self.history_length = 10
        self.cwnds = []
        self.losses = []

    def add_monitor_interval(self, monitor_interval):
        if len(self.cwnds) < self.history_length:
            ## haven't seen a full history length yet ##
            self.cwnds.append(monitor_interval.cwnd)
            self.losses.append(compute_reward(monitor_interval))
        else:
            # clear out the oldest interval
            self.cwnds = self.cwnds[1:]
            self.losses = self.losses[1:]

            # append the new one
            self.cwnds.append(monitor_interval.cwnd)
            self.losses.append(compute_reward(monitor_interval))
