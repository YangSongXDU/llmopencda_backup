# -*- coding: utf-8 -*-
"""Lightweight process and host resource sampling for experiment CSVs."""

import os
import subprocess


class ProcessResourceMonitor(object):
    """Sample client-process CPU/RSS and optional host GPU utilization."""

    def __init__(self, config=None):
        config = config or {}
        self.enabled = bool(config.get('enabled', False))
        self.sample_gpu = bool(config.get('sample_gpu', True))
        self.gpu_index = int(config.get('gpu_index', 0))
        self.gpu_sample_interval_steps = max(
            1, int(config.get('gpu_sample_interval_steps', 20)))
        self.nvidia_smi_command = str(config.get(
            'nvidia_smi_command', 'nvidia-smi'))
        self.process = None
        self.psutil = None
        self.nvml = None
        self.gpu_handle = None
        self._nvml_initialized = False
        self._sample_count = 0
        self._last_gpu_utilization = -1.0
        self._last_gpu_memory_used_mb = -1.0

        if not self.enabled:
            return

        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            self.psutil = None
            self.process = None

        if self.sample_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                    self.gpu_index)
                self._nvml_initialized = True
            except Exception:
                self.nvml = None
                self.gpu_handle = None

    @staticmethod
    def _empty_sample():
        return {
            'client_process_cpu_percent': -1.0,
            'client_process_rss_mb': -1.0,
            'host_cpu_percent': -1.0,
            'host_memory_used_mb': -1.0,
            'gpu_utilization_percent': -1.0,
            'gpu_memory_used_mb': -1.0
        }

    def sample(self):
        values = self._empty_sample()
        if not self.enabled:
            return values

        if self.process is not None and self.psutil is not None:
            try:
                values['client_process_cpu_percent'] = float(
                    self.process.cpu_percent(interval=None))
                values['client_process_rss_mb'] = float(
                    self.process.memory_info().rss) / (1024.0 * 1024.0)
                values['host_cpu_percent'] = float(
                    self.psutil.cpu_percent(interval=None))
                values['host_memory_used_mb'] = float(
                    self.psutil.virtual_memory().used) / (1024.0 * 1024.0)
            except Exception:
                pass

        self._sample_count += 1
        if self.gpu_handle is not None and self.nvml is not None:
            try:
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(
                    self.gpu_handle)
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                values['gpu_utilization_percent'] = float(utilization.gpu)
                values['gpu_memory_used_mb'] = float(memory.used) / (
                    1024.0 * 1024.0)
                self._last_gpu_utilization = values[
                    'gpu_utilization_percent']
                self._last_gpu_memory_used_mb = values[
                    'gpu_memory_used_mb']
            except Exception:
                pass
        elif self.sample_gpu:
            should_refresh = (
                self._last_gpu_utilization < 0.0 or
                (self._sample_count - 1) % self.gpu_sample_interval_steps == 0)
            if should_refresh:
                try:
                    output = subprocess.check_output([
                        self.nvidia_smi_command,
                        '--id=%d' % self.gpu_index,
                        '--query-gpu=utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'
                    ], stderr=subprocess.DEVNULL, timeout=1.0,
                        universal_newlines=True)
                    first_line = output.strip().splitlines()[0]
                    utilization, memory_used = first_line.split(',')[:2]
                    self._last_gpu_utilization = float(utilization.strip())
                    self._last_gpu_memory_used_mb = float(memory_used.strip())
                except Exception:
                    pass
            values['gpu_utilization_percent'] = self._last_gpu_utilization
            values['gpu_memory_used_mb'] = self._last_gpu_memory_used_mb
        return values

    def close(self):
        if self._nvml_initialized and self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml_initialized = False
