import pandas as pd
import json as json 

class LVLogger:
    def __init__(self, route_info):
        self.route_info = route_info
        
        # Extract commonly used fields for easier access
        self.route_id = route_info.get('route_id')
        self.route_long_name = route_info.get('route_name')  # Note: route_info uses 'route_name'
        self.route_short_name = route_info.get('route_short_name')
        
        self.stop_topology_logs = {}
        self.direction_topology_logs = {}
        self.performance_logs = {}
        self.navigation_structures = {}
        self.set_up_log_structure()
    
    def set_up_log_structure(self):
        # ===== STOP TOPOLOGY VALIDATION LOGS =====
        self.stop_topology_logs = {
            'parent_station_labels': {}, # Classification of Parent station if possible
            'parent_station_violations': {},
            'stop_id_labels':{},
            'stop_id_violations':{},
            'metadata': {
                'total_violations': 0,
                'violation_counts_by_type': {
                    'parent_station': 0, 
                    'stop_id': 0
                },
                'route_summary': {
                    'total_stops': 0,
                    'total_parent_stations': 0,
                    'topology_types': {}  # Count of each topology type
                }
            }
        }
        
        # ===== DIRECTION TOPOLOGY VALIDATION LOGS =====
        self.direction_topology_logs = {
            'direction_labels': {},     # Classification of Direction if possible
            'direction_violations': {},     # Individual stop_id assignment issues 
            'stop_id_labels':{},
            'stop_id_violations':{},
            'metadata': {
                'canonical_patterns':{},
                'total_violations': 0,
                'violation_counts_by_type': {
                    'direction': 0,
                    'stop_id': 0
                },
                'route_summary': {
                    'total_directions': 0,
                    'topology_types': {} # Count of each topology type
                }
            }
        }
        
        # ===== PERFORMANCE & RELIABILITY ANALYSIS LOGS =====
        self.performance_logs = {
            'punctuality_barcharts': {},   
            'stops_regulatory_labels': {},
            'regulatory_violations' : {},
            'regulatory_performance_violations':{},
            'histograms_stops': {}, 
            'travel_times':{},        
            'metadata': {
                'analysis_period': {
                    'start_date': None,
                    'end_date': None,
                    'total_days': 0
                },
                'performance_summary': {
                    'overall_on_time_rate': 0.0,
                    'average_delay': 0.0,
                    'reliability_score': 0.0
                }
            }
        }
        
        # ===== NAVIGATION STRUCTURES =====
        self.navigation_structures = {
            'stop_hierarchies': {
                'by_stop_name': {},    # stop_name -> parent_station -> stop_ids
                'by_stop_id': {},      # stop_id -> (parent_station, stop_name, label)
            },
            'direction_hierarchies': {
                'by_direction': {},    # direction_id -> pattern -> stop_ids
                'by_stop_id': {}       #  stop_ids -> pattern ->  direction_id
            }
        }
    
    def store_labels_and_violations(self, domain: str, labels: dict, violations: dict):
        """
        Store both labels and violations cleanly.
        """
        logs = {
            'stop_topology': self.stop_topology_logs,
            'direction_topology': self.direction_topology_logs
        }[domain]

        for label_type, label_data in labels.items():
            if f"{label_type}_labels" not in logs:
                logs[f"{label_type}_labels"] = {}
            logs[f"{label_type}_labels"].update({
                f"{label_type}_{i + len(logs[f'{label_type}_labels'])}": v for i, v in enumerate(label_data)
            })

        for violation_type, violation_data in violations.items():
            if f"{violation_type}_violations" not in logs:
                logs[f"{violation_type}_violations"] = {}
            logs[f"{violation_type}_violations"].update({
                f"{violation_type}_{i + len(logs[f'{violation_type}_violations'])}": v for i, v in enumerate(violation_data)
            })

        logs['metadata']['labels_counts_by_type'] = {
            k: len(labels.get(k, [])) for k in labels
        }
        logs['metadata']['total_labels'] = sum(len(vs) for vs in labels.values())
        logs['metadata']['violation_counts_by_type'] = {
            k: len(violations.get(k, [])) for k in violations
        }
        logs['metadata']['total_violations'] = sum(len(vs) for vs in violations.values())

    def create_violation_entry(self, violation_type, severity, description, **details):
        """Create standardized violation entry with searchable fields"""
        return {
            'violation_type': violation_type,
            'severity': severity,
            'description': description,
            
            # SEARCHABLE FIELDS - always included
            'route_id': str(self.route_id),
            'route_name': self.route_long_name,
            'route_short_name': self.route_short_name,
       
            # Original details
            **details
        }
    
    def add_violation(self, domain: str, violation_type: str, violation_entry: dict):
        """
        Add a single violation entry to the specified domain and type.
        """
        # Input validation
        if domain not in ['stop_topology', 'direction_topology']:
            raise ValueError(f"Invalid domain: {domain}. Must be 'stop_topology' or 'direction_topology'")
        
        if not isinstance(violation_entry, dict):
            raise TypeError("violation_entry must be a dictionary")
            
        logs = {
            'stop_topology': self.stop_topology_logs,
            'direction_topology': self.direction_topology_logs
        }[domain]

        # Ensure violation type exists in logs
        violations_key = f"{violation_type}_violations"
        if violations_key not in logs:
            logs[violations_key] = {}
            # Initialize count if not exists
            if violation_type not in logs['metadata']['violation_counts_by_type']:
                logs['metadata']['violation_counts_by_type'][violation_type] = 0

        key = f"{violation_type}_{len(logs[violations_key])}"
        logs[violations_key][key] = violation_entry
        logs['metadata']['violation_counts_by_type'][violation_type] += 1
        logs['metadata']['total_violations'] += 1
        return violation_entry

    def add_label(self, domain: str, label_type: str, entity_key: str, label_value: str):
        """
        Add a single label entry to the specified domain and type.
        """
        # Input validation
        if domain not in ['stop_topology', 'direction_topology']:
            raise ValueError(f"Invalid domain: {domain}. Must be 'stop_topology' or 'direction_topology'")
            
        logs = {
            'stop_topology': self.stop_topology_logs,
            'direction_topology': self.direction_topology_logs
        }[domain]

        labels_key = f"{label_type}_labels"
        if labels_key not in logs:
            logs[labels_key] = {}

        logs[labels_key][entity_key] = label_value

        if 'labels_counts_by_type' not in logs['metadata']:
            logs['metadata']['labels_counts_by_type'] = {}

        logs['metadata']['labels_counts_by_type'].setdefault(label_type, 0)
        logs['metadata']['labels_counts_by_type'][label_type] += 1
        
        # Recalculate total labels
        logs['metadata']['total_labels'] = sum(logs['metadata']['labels_counts_by_type'].values())

        return label_value