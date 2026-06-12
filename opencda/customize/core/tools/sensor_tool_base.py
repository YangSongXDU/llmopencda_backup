# -*- coding: utf-8 -*-
"""
Base classes for sensor tools used by the LLM Sensor Agent.

Each tool receives a context dictionary and returns a ToolResult.
The context is intentionally lightweight so that tools can be called
from planning agents without changing OpenCDA's core controller API.
"""


class ToolResult(object):
    """
    Lightweight result object returned by every sensor tool.
    """

    def __init__(self, tool_name, success=True, data=None, cost=0.0, reason=''):
        self.tool_name = tool_name
        self.success = success
        self.data = data if data is not None else {}
        self.cost = float(cost)
        self.reason = reason

    def to_dict(self):
        result = {
            'tool_name': self.tool_name,
            'success': self.success,
            'cost': self.cost,
            'reason': self.reason
        }
        result.update(self.data)
        return result


class SensorToolBase(object):
    """
    Base class for all sensor tools.
    """

    def __init__(self, tool_name, cost=1.0, enabled=True):
        self.tool_name = tool_name
        self.cost = float(cost)
        self.enabled = bool(enabled)

    def run(self, context):
        raise NotImplementedError

    def disabled_result(self):
        return ToolResult(
            tool_name=self.tool_name,
            success=False,
            data={},
            cost=0.0,
            reason='Tool disabled.'
        )
