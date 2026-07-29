#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot resource-aware LLM Agent tool-selection results."""

import argparse
import csv
import os


DEFAULT_CSV = (
    'opencda_output/agent_single_llm_tool_demo/'
    'agent_single_llm_tool_demo.csv')


def _as_bool(value):
    return str(value).strip().lower() in ['true', '1', 'yes']


def _as_float(value, default=0.0):
    try:
        if value is None or value == '':
            return default
        return float(value)
    except Exception:
        return default


def _load_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _encode(values):
    labels = []
    encoded = []
    for value in values:
        if value not in labels:
            labels.append(value)
        encoded.append(labels.index(value))
    return encoded, labels


def _save(fig, path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=DEFAULT_CSV)
    parser.add_argument('--out-dir', default='results/agent_tool_selection')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = _load_rows(args.csv)
    steps = [_as_float(row.get('step')) for row in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([_as_float(row.get('ego_x')) for row in rows],
            [_as_float(row.get('ego_y')) for row in rows], linewidth=2)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Ego trajectory')
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(args.out_dir, 'ego_trajectory.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    tool_series = [
            ('camera_called', 'Camera'),
            ('radar_called', 'Radar'),
            ('lidar_called', 'LiDAR'),
            ('fusion_called', 'Fusion')]
    event_steps = [
        [steps[index] for index, row in enumerate(rows)
         if _as_bool(row.get(field))]
        for field, _ in tool_series]
    ax.eventplot(event_steps, lineoffsets=range(len(tool_series)),
                 linelengths=0.7, linewidths=0.8)
    ax.set_xlabel('Step')
    ax.set_title('Selected tool calls')
    ax.set_yticks(range(len(tool_series)))
    ax.set_yticklabels([label for _, label in tool_series])
    ax.set_ylim(-0.6, len(tool_series) - 0.4)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(args.out_dir, 'tool_calls_timeline.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    cumulative = []
    total = 0.0
    for row in rows:
        total += _as_float(row.get('tool_total_cost'))
        cumulative.append(total)
    ax.plot(steps, cumulative, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative cost')
    ax.set_title('Cumulative tool cost')
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(args.out_dir, 'cumulative_tool_cost.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(steps, [_as_float(row.get('tool_total_cost')) for row in rows],
            linewidth=1.2, label='Counted tool cost')
    if rows and 'oracle_tool_cost' in rows[0]:
        ax.plot(steps, [_as_float(row.get('oracle_tool_cost')) for row in rows],
                linewidth=1.0, alpha=0.8, label='Prototype oracle cost')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cost [proxy units]')
    ax.set_title('Per-step tool cost')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(args.out_dir, 'tool_cost_timeline.png'))
    plt.close(fig)

    for field, title, filename in [
            ('risk_level', 'Risk level timeline', 'risk_timeline.png'),
            ('uncertainty_level', 'Uncertainty timeline',
             'uncertainty_timeline.png'),
            ('overtake_state', 'Overtake state timeline',
             'overtake_state_timeline.png'),
            ('llm_backend', 'LLM backend timeline',
             'llm_backend_timeline.png')]:
        encoded, labels = _encode([row.get(field, '') for row in rows])
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        ax.step(steps, encoded, where='post', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(args.out_dir, filename))
        plt.close(fig)

    if rows and ('overtake_attempt_count' in rows[0] or
                 'completed_overtake_count' in rows[0]):
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        ax.step(
            steps,
            [_as_float(row.get('overtake_attempt_count')) for row in rows],
            where='post', label='Attempts')
        ax.step(
            steps,
            [_as_float(row.get('completed_overtake_count')) for row in rows],
            where='post', label='Completed')
        if 'overtake_abort_count' in rows[0]:
            ax.step(
                steps,
                [_as_float(row.get('overtake_abort_count')) for row in rows],
                where='post', label='Aborted')
        ax.set_xlabel('Step')
        ax.set_ylabel('Count')
        ax.set_title('Repeated overtaking progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(
            args.out_dir, 'overtake_count_timeline.png'))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for field, label in [
            ('lidar_front_distance', 'LiDAR'),
            ('radar_front_distance', 'Radar'),
            ('fusion_front_distance', 'Fusion')]:
        values = [_as_float(row.get(field), 999.0) for row in rows]
        values = [v if v < 200.0 else None for v in values]
        ax.plot(steps, values, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance [m]')
    ax.set_title('Front distance estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(args.out_dir, 'distance_comparison.png'))
    plt.close(fig)

    if rows and 'llm_call_executed' in rows[0]:
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        call_steps = [steps[index] for index, row in enumerate(rows)
                      if _as_bool(row.get('llm_call_executed'))]
        fallback_steps = [steps[index] for index, row in enumerate(rows)
                          if _as_bool(row.get('llm_call_executed')) and
                          _as_bool(row.get('llm_fallback_used'))]
        retry_steps = [steps[index] for index, row in enumerate(rows)
                       if _as_float(row.get('llm_retry_count')) > 0]
        ax.eventplot([call_steps, retry_steps, fallback_steps],
                     lineoffsets=[0, 1, 2], linelengths=0.7,
                     linewidths=1.0)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['LLM call', 'Retried', 'Fallback'])
        ax.set_xlabel('Step')
        ax.set_title('LLM request outcomes')
        ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(args.out_dir, 'llm_request_timeline.png'))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        latency = [
            _as_float(row.get('llm_request_latency_ms'))
            if _as_bool(row.get('llm_call_executed')) else None
            for row in rows]
        ax.plot(steps, latency, marker='.', linewidth=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Latency [ms]')
        ax.set_title('LLM request latency')
        ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(args.out_dir, 'llm_latency_timeline.png'))
        plt.close(fig)

    if rows and 'client_process_cpu_percent' in rows[0]:
        resource_fields = [
            ('client_process_cpu_percent', 'Client CPU [%]'),
            ('client_process_rss_mb', 'Client RSS [MB]'),
            ('gpu_utilization_percent', 'GPU utilization [%]')]
        if any(any(_as_float(row.get(field), -1.0) >= 0.0 for row in rows)
               for field, _ in resource_fields):
            fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
            for ax, (field, label) in zip(axes, resource_fields):
                values = [_as_float(row.get(field), -1.0) for row in rows]
                values = [value if value >= 0.0 else None for value in values]
                ax.plot(steps, values, linewidth=0.9)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel('Step')
            axes[0].set_title('Experiment resource utilization')
            _save(fig, os.path.join(
                args.out_dir, 'resource_utilization_timeline.png'))
            plt.close(fig)

    if rows and 'current_lane_continuous_distance' in rows[0]:
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        for field, label in [
                ('current_lane_continuous_distance', 'Current lane'),
                ('left_lane_continuous_distance', 'Left lane'),
                ('right_lane_continuous_distance', 'Right lane')]:
            ax.plot(steps, [_as_float(row.get(field)) for row in rows],
                    linewidth=1.0, label=label)
        ax.set_xlabel('Step')
        ax.set_ylabel('Continuous distance [m]')
        ax.set_title('Lane continuity evidence')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(args.out_dir, 'lane_continuity_timeline.png'))
        plt.close(fig)

    print('Plots saved to %s' % args.out_dir)


if __name__ == '__main__':
    main()
