#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze resource-aware LLM Agent tool-selection CSV logs."""

import argparse
import csv
import json
import math
import os
from collections import Counter


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


def _split_tools(value):
    if not value:
        return []
    return [item for item in str(value).split('|') if item]


def _distribution(rows, field):
    counter = Counter(row.get(field, '') for row in rows)
    if '' in counter:
        del counter['']
    return dict(counter)


def _ratio(rows, field):
    if not rows:
        return 0.0
    return sum(1 for row in rows if _as_bool(row.get(field, ''))) / float(len(rows))


def _mean(rows, field):
    if not rows:
        return 0.0
    return sum(_as_float(row.get(field, 0.0)) for row in rows) / float(len(rows))


def _available_values(rows, field):
    values = []
    for row in rows:
        if field not in row or row.get(field, '') == '':
            continue
        value = _as_float(row.get(field), -1.0)
        if value >= 0.0:
            values.append(value)
    return values


def _value_mean(rows, field):
    values = _available_values(rows, field)
    return sum(values) / float(len(values)) if values else None


def _percentile(rows, field, percentile):
    values = sorted(_available_values(rows, field))
    if not values:
        return None
    index = int(round((len(values) - 1) * float(percentile) / 100.0))
    return values[max(0, min(index, len(values) - 1))]


def _travel_distance(rows):
    total = 0.0
    prev = None
    for row in rows:
        point = (_as_float(row.get('ego_x')), _as_float(row.get('ego_y')))
        if prev is not None:
            total += math.sqrt((point[0] - prev[0]) ** 2 +
                               (point[1] - prev[1]) ** 2)
        prev = point
    return total


def _first_step(rows, field, value):
    for row in rows:
        if row.get(field, '') == value:
            return int(_as_float(row.get('step'), -1))
    return None


def _event_steps(rows, field, value):
    return [int(_as_float(row.get('step'), -1)) for row in rows
            if row.get(field, '') == value]


def _max_int(rows, field):
    if not rows:
        return 0
    return int(max(_as_float(row.get(field), 0.0) for row in rows))


def analyze_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    tool_counter = Counter()
    requested_counter = Counter()
    skipped_counter = Counter()
    cached_counter = Counter()
    for row in rows:
        tool_counter.update(_split_tools(row.get('executed_tools') or
                                         row.get('called_tools', '')))
        requested_counter.update(_split_tools(row.get('requested_tools', '')))
        skipped_counter.update(_split_tools(row.get('skipped_tools', '')))
        cached_counter.update(_split_tools(row.get('cached_tools', '')))

    has_call_marker = bool(rows and 'llm_call_executed' in rows[0])
    llm_call_rows = [row for row in rows
                     if _as_bool(row.get('llm_call_executed', ''))] \
        if has_call_marker else rows

    summary = {
        'csv': path,
        'rows': len(rows),
        'llm_metric_scope': 'llm_calls' if has_call_marker else 'legacy_frames',
        'llm_call_count': len(llm_call_rows) if has_call_marker else None,
        'llm_backend_distribution': _distribution(
            llm_call_rows, 'llm_backend'),
        'fallback_ratio': _ratio(llm_call_rows, 'llm_fallback_used'),
        'fallback_ratio_all_frames': _ratio(rows, 'llm_fallback_used'),
        'llm_call_success_ratio': _ratio(llm_call_rows, 'llm_call_success')
        if has_call_marker else None,
        'llm_response_valid_ratio': _ratio(
            llm_call_rows, 'llm_response_valid') if has_call_marker else None,
        'llm_retry_distribution': _distribution(
            llm_call_rows, 'llm_retry_count'),
        'llm_http_status_distribution': _distribution(
            llm_call_rows, 'llm_http_status'),
        'llm_fallback_reason_distribution': _distribution(
            llm_call_rows, 'llm_fallback_reason'),
        'mean_llm_request_latency_ms': _value_mean(
            llm_call_rows, 'llm_request_latency_ms'),
        'p95_llm_request_latency_ms': _percentile(
            llm_call_rows, 'llm_request_latency_ms', 95),
        'llm_circuit_open_ratio': _ratio(
            llm_call_rows, 'llm_circuit_open') if has_call_marker else None,
        'risk_level_distribution': _distribution(rows, 'risk_level'),
        'uncertainty_level_distribution': _distribution(rows, 'uncertainty_level'),
        'resource_budget_level_distribution': _distribution(
            rows, 'resource_budget_level'),
        'fusion_called_ratio': _ratio(rows, 'fusion_called'),
        'camera_called_ratio': _ratio(rows, 'camera_called'),
        'lidar_called_ratio': _ratio(rows, 'lidar_called'),
        'radar_called_ratio': _ratio(rows, 'radar_called'),
        'front_debug_called_ratio': _ratio(rows, 'front_debug_called'),
        'lane_check_called_ratio': _ratio(rows, 'lane_check_called'),
        'mean_tool_call_count': _mean(rows, 'tool_call_count'),
        'mean_resource_counted_tool_count': _value_mean(
            rows, 'resource_counted_tool_count'),
        'mean_tool_total_cost': _mean(rows, 'tool_total_cost'),
        'mean_tool_budget_used': _mean(rows, 'tool_budget_used'),
        'mean_total_tool_runtime_ms': _mean(rows, 'total_tool_runtime_ms'),
        'p95_total_tool_runtime_ms': _percentile(
            rows, 'total_tool_runtime_ms', 95),
        'mean_agent_cycle_runtime_ms': _value_mean(
            rows, 'agent_cycle_runtime_ms'),
        'p95_agent_cycle_runtime_ms': _percentile(
            rows, 'agent_cycle_runtime_ms', 95),
        'mean_oracle_tool_runtime_ms': _value_mean(
            rows, 'oracle_tool_runtime_ms'),
        'mean_oracle_tool_cost': _value_mean(rows, 'oracle_tool_cost'),
        'mean_client_process_cpu_percent': _value_mean(
            rows, 'client_process_cpu_percent'),
        'mean_client_process_rss_mb': _value_mean(
            rows, 'client_process_rss_mb'),
        'mean_host_cpu_percent': _value_mean(rows, 'host_cpu_percent'),
        'mean_host_memory_used_mb': _value_mean(
            rows, 'host_memory_used_mb'),
        'mean_gpu_utilization_percent': _value_mean(
            rows, 'gpu_utilization_percent'),
        'mean_gpu_memory_used_mb': _value_mean(
            rows, 'gpu_memory_used_mb'),
        'tool_execution_counts': dict(tool_counter),
        'requested_tool_counts': dict(requested_counter),
        'skipped_tool_counts': dict(skipped_counter),
        'cached_tool_counts': dict(cached_counter),
        'tool_selection_reason_top10': Counter(
            row.get('tool_selection_reason', '') for row in rows
            if row.get('tool_selection_reason', '')).most_common(10),
        'fusion_trigger_reason_top10': Counter(
            row.get('fusion_trigger_reason', '') for row in rows
            if row.get('fusion_trigger_reason', '')).most_common(10),
        'safety_evidence_distribution': _distribution(rows, 'safety_evidence'),
        'overtake_state_distribution': _distribution(rows, 'overtake_state'),
        'first_overtake_left_step': _first_step(
            rows, 'maneuver_applied', 'overtake_left'),
        'first_return_to_original_lane_step': _first_step(
            rows, 'maneuver_applied', 'return_to_original_lane'),
        'first_overtake_done_step': _first_step(
            rows, 'maneuver_applied', 'overtake_done'),
        'overtake_left_steps': _event_steps(
            rows, 'maneuver_applied', 'overtake_left'),
        'return_to_original_lane_steps': _event_steps(
            rows, 'maneuver_applied', 'return_to_original_lane'),
        'overtake_done_steps': _event_steps(
            rows, 'maneuver_applied', 'overtake_done'),
        'overtake_attempt_count': _max_int(
            rows, 'overtake_attempt_count'),
        'completed_overtake_count': _max_int(
            rows, 'completed_overtake_count'),
        'overtake_abort_count': _max_int(rows, 'overtake_abort_count'),
        'overtake_abort_steps': _event_steps(
            rows, 'maneuver_applied', 'abort_overtake'),
        'overtake_abort_reason_distribution': _distribution(
            rows, 'last_overtake_abort_reason'),
        'travel_distance': _travel_distance(rows)
    }
    summary['overtake_done_observed'] = (
        summary['first_overtake_done_step'] is not None or
        'OVERTAKE_DONE' in summary['overtake_state_distribution'])
    return summary


def write_markdown(summary, path):
    lines = [
        '# Agent Tool Selection CSV Summary',
        '',
        '| Metric | Value |',
        '|---|---:|',
        '| Rows | %d |' % summary['rows'],
        '| Fusion called ratio | %.4f |' % summary['fusion_called_ratio'],
        '| Camera called ratio | %.4f |' % summary['camera_called_ratio'],
        '| LiDAR called ratio | %.4f |' % summary['lidar_called_ratio'],
        '| Radar called ratio | %.4f |' % summary['radar_called_ratio'],
        '| Mean tool call count | %.4f |' % summary['mean_tool_call_count'],
        '| Mean resource-counted tool calls | %s |' % (
            '%.4f' % summary['mean_resource_counted_tool_count']
            if summary['mean_resource_counted_tool_count'] is not None
            else 'not recorded in legacy CSV'),
        '| Mean tool total cost | %.4f |' % summary['mean_tool_total_cost'],
        '| Mean runtime ms | %.4f |' % summary['mean_total_tool_runtime_ms'],
        '| LLM metric scope | %s |' % summary['llm_metric_scope'],
        '| LLM calls | %s |' % (
            summary['llm_call_count']
            if summary['llm_call_count'] is not None
            else 'not recorded in legacy CSV'),
        '| Fallback ratio | %.4f |' % summary['fallback_ratio'],
        '| Travel distance | %.4f |' % summary['travel_distance'],
        '| Overtake done observed | %s |' %
        summary['overtake_done_observed'],
        '| Overtake attempts | %d |' % summary['overtake_attempt_count'],
        '| Completed overtakes | %d |' % summary['completed_overtake_count'],
        '| Aborted overtakes | %d |' % summary['overtake_abort_count'],
        '',
        '## Tool Execution Counts',
        '',
        json.dumps(summary['tool_execution_counts'], indent=2, sort_keys=True),
        '',
        '## Requested Tool Counts',
        '',
        json.dumps(summary['requested_tool_counts'], indent=2, sort_keys=True),
        '',
        '## Risk Distribution',
        '',
        json.dumps(summary['risk_level_distribution'], indent=2, sort_keys=True),
        ''
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=DEFAULT_CSV)
    parser.add_argument('--out-json',
                        default='results/agent_tool_selection_summary.json')
    parser.add_argument('--out-md',
                        default='results/agent_tool_selection_summary.md')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    summary = analyze_csv(args.csv)
    out_dir = os.path.dirname(args.out_json)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    md_dir = os.path.dirname(args.out_md)
    if md_dir and not os.path.exists(md_dir):
        os.makedirs(md_dir)

    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_markdown(summary, args.out_md)
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
