
class LVLogger:
    def __init__(self, route_info):
        self.route_info = route_info
        
        # Extract commonly used fields for easier access
        self.route_id = route_info.get('route_id')
        self.route_long_name = route_info.get('route_long_name')
        self.route_short_name = route_info.get('route_short_name')
        
        self.stop_topology_logs = {}
        self.direction_topology_logs = {}
        self.performance_logs = {}
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
                'violation_counts_by_severity': {},   # ← new
                'route_summary': {
                    'total_parent_stations': 0,
                    'total_stop_ids': 0
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
                'violation_counts_by_severity': {},   # ← new
                'route_summary': {
                    'total_directions': 0
                }
            }
        }
        
        # ===== REGULATORY STOPS LOGS =====
        self.regulatory_stops_logs = {
            'stop_id_regulatory_labels': {},
            'metadata': {
                'total_regulatory_stops': 0
            }
        }

        # ===== PERFORMANCE & RELIABILITY ANALYSIS LOGS =====
        self.performance_logs = {
            'analytics_logs': {}, 
            'travel_times':{},        
            'metadata': {
                'performance_summary': {
                    'overall_too_early_rate': 0.0,
                    'overall_on_time_rate': 0.0,
                    'overall_too_late_rate':0.0,
                    'average_departure_delay': 0.0
                }
            }
        }
    
    def get_logs(self, domain: str):
        valid_domains = {
            'stop_topology': self.stop_topology_logs,
            'direction_topology': self.direction_topology_logs,
            'regulatory_stops': self.regulatory_stops_logs,
            'performance': self.performance_logs
        }
        if domain not in valid_domains:
            raise ValueError(f"Invalid domain: {domain}. Must be one of {list(valid_domains.keys())}")
        return valid_domains[domain]

    def build_entity_key(self, domain: str, *,
                         stop_id: str = None,
                         direction_id: str = None,
                         parent_station: str = None,
                         time_type: str = None,
                         from_stop_id: str = None,
                         to_stop_id: str = None) -> str:
        """
        Construct a consistent entity key string in the format:
        {domain}_{route_id}_{kind}_{identifier}

        Logs all input parameters and includes them in the exception
        if none of the patterns match.
        """
        # --- Most specific compound keys first ---
        if from_stop_id and to_stop_id and direction_id and time_type:
            kind       = "segment"
            identifier = f"{direction_id}_{from_stop_id}_{to_stop_id}_{time_type}"

        elif stop_id and direction_id and time_type:
            kind       = "direction_id_stop_id_time_type"
            identifier = f"{direction_id}_{stop_id}_{time_type}"

        elif parent_station and stop_id:
            kind       = "parent_station_stop_id"
            identifier = f"{parent_station}_{stop_id}"

        elif stop_id and direction_id:
            kind       = "direction_id_stop_id"
            identifier = f"{direction_id}_{stop_id}"

        elif stop_id:
            kind       = "stop_id"
            identifier = str(stop_id)

        elif direction_id:
            kind       = "direction_id"
            identifier = str(direction_id)

        elif parent_station:
            kind       = "parent_station"
            identifier = str(parent_station)

        else:
            # No valid combination—raise with full context
            raise ValueError(
                f"Insufficient data to build entity key for domain={domain!r}, "
                f"route_id={self.route_id!r}, direction_id={direction_id!r}, "
                f"stop_id={stop_id!r}, parent_station={parent_station!r}, "
                f"time_type={time_type!r}, from_stop_id={from_stop_id!r}, "
                f"to_stop_id={to_stop_id!r}"
            )

        return f"{domain}_{self.route_id}_{kind}_{identifier}"

    def create_violation_entry(self, violation_type, severity, description, entity_key, **details):
        """
        Create standardized violation entry with searchable and descriptive fields.
        Includes entity_key as part of the metadata.
        """
        return {
            'violation_type': violation_type,
            'severity': severity,
            'description': description,
            'entity_key': str(entity_key),  

            'route_id': str(self.route_id),
            'route_long_name': self.route_long_name,
            'route_short_name': self.route_short_name,
            **details
        }

    def create_label_entry(self, label_type, description, entity_key, **details):
        """
        Create standardized label entry with searchable and descriptive fields.
        Includes entity_key as part of the metadata.
        """
        return {
            'label_type': label_type,
            'description': description,
            'entity_key': str(entity_key),  
            
            'route_id': str(self.route_id),
            'route_long_name': self.route_long_name,
            'route_short_name': self.route_short_name,
            **details
        }

    def add_violation(self, domain: str, violation_type: str, entity_key: str, violation_entry: dict):
        """
        Add a single violation entry to the specified domain and type, using entity_key as the key.
        """
        if domain not in ['stop_topology', 'direction_topology', 'regulatory_stops', 'performance']:
            raise ValueError(f"Invalid domain: {domain}. Must be one of the supported domains.")

        if not isinstance(violation_entry, dict):
            raise TypeError("violation_entry must be a dictionary.")

        if not isinstance(entity_key, str):
            entity_key = str(entity_key)

        logs = self.get_logs(domain)

        violations_key = f"{violation_type}_violations"
        if violations_key not in logs:
            logs[violations_key] = {}

        logs[violations_key][entity_key] = violation_entry

        # Update metadata
        logs.setdefault('metadata', {})
        logs['metadata'].setdefault('violation_counts_by_type', {})
        logs['metadata']['violation_counts_by_type'][violation_type] = logs['metadata']['violation_counts_by_type'].get(violation_type, 0) + 1
        logs['metadata']['total_violations'] = sum(logs['metadata']['violation_counts_by_type'].values())

        md = logs['metadata']
        sev = violation_entry.get('severity', 0)
        md.setdefault('violation_counts_by_severity', {})
        md['violation_counts_by_severity'][sev] = (md['violation_counts_by_severity'].get(sev, 0) + 1)

        return violation_entry

    def add_label(self, domain: str, label_type: str, entity_key: str, label_entry: dict):
        """
        Add a single label entry to the specified domain and type.

        Args:
            domain: one of ['stop_topology', 'direction_topology', 'regulatory_stops', 'performance']
            label_type: like 'stop_id', 'parent_station', 'direction', etc.
            entity_key: the ID of the thing being labeled (e.g., stop_id, parent_station name)
            label_entry: the full label metadata (e.g. created via `create_label_entry`)
        """
        if domain not in ['stop_topology', 'direction_topology', 'regulatory_stops', 'performance']:
            raise ValueError(f"Invalid domain: {domain}. Must be one of the supported domains.")

        if not isinstance(label_entry, dict):
            raise TypeError("label_entry must be a dictionary.")

        if not isinstance(entity_key, str):
            entity_key = str(entity_key)

        logs = self.get_logs(domain)

        # 1) Insert the label
        labels_key = f"{label_type}_labels"
        logs.setdefault(labels_key, {})
        logs[labels_key][entity_key] = label_entry

        # 2) Update generic per-type counts
        logs.setdefault('metadata', {})
        md = logs['metadata']
        md.setdefault('labels_counts_by_type', {})
        md['labels_counts_by_type'][label_type] = (
            md['labels_counts_by_type'].get(label_type, 0) + 1
        )

        # 3) Maintain the correct total counter
        if domain == 'regulatory_stops':
            # single label-type; count each labeled stop
            md['total_regulatory_stops'] = md.get('total_regulatory_stops', 0) + 1
        else:
            # generic label domains
            md['total_labels'] = sum(md['labels_counts_by_type'].values())

        # 4) Update our topology route-summary totals
        if domain == 'stop_topology':
            rs = md.setdefault('route_summary', {})
            # how many unique parent_station labels
            rs['total_parent_stations'] = len(logs.get('parent_station_labels', {}))
            # how many unique stop_id labels
            rs['total_stop_ids']      = len(logs.get('stop_id_labels', {}))

        elif domain == 'direction_topology':
            rs = md.setdefault('route_summary', {})
            # how many unique direction labels
            rs['total_directions'] = len(logs.get('direction_labels', {}))

        return label_entry
    
    def get_available_keys(self, domain: str) -> dict:
        """
        Get all available keys for a domain.
        
        Returns:
            dict: {log_type: [keys]} for all log types in the domain
            
        Example:
            For stop_topology domain, returns:
            {
                'parent_station_labels': ['stop_topology_123_parent_station_Central', ...],
                'parent_station_violations': ['stop_topology_123_parent_station_Central_violation', ...],
                'stop_id_labels': ['stop_topology_123_stop_id_456', ...],
                'stop_id_violations': ['stop_topology_123_stop_id_789_violation', ...],
            }
        """
        logs = self.get_logs(domain)
        result = {}
        
        # Go through all log types in this domain
        for log_type, log_data in logs.items():  
            # Get all keys from this log type
            result[log_type] = list(log_data.keys())
        
        return result

    def get_all_keys_regex(self, domain: str, *contains_parts, match_all=True, log_type=None, exclude=None) -> list:
        """
        Get all keys from a domain with regex support, log type filtering, and exclusions.
        
        Args:
            domain: Domain to search in ('stop_topology', 'direction_topology', 'regulatory_stops', 'performance')
            *contains_parts: Parts to search for (can include regex patterns)
            match_all: If True, ALL parts must match. If False, ANY part can match.
            log_type: Filter by specific log type - e.g. 'parent_station_labels', 'direction_violations', etc., or None for all
            exclude: Parts that must NOT be in the key. Can be string, list, or None
        """
        import re
        
        domain_keys = self.get_available_keys(domain)
        matching_keys = []
        
        # Compile inclusion patterns
        patterns = [re.compile(str(part)) for part in contains_parts]
        
        # Compile exclusion patterns
        exclusion_patterns = []
        if exclude is not None:
            if isinstance(exclude, (list, tuple)):
                exclusion_patterns = [re.compile(str(part)) for part in exclude]
            else:
                exclusion_patterns = [re.compile(str(exclude))]
        
        def key_matches_criteria(key):
            """Check if key matches inclusion criteria and doesn't match exclusion criteria."""
            
            # Check inclusion patterns
            if match_all:
                # ALL inclusion patterns must match
                if not all(pattern.search(key) for pattern in patterns):
                    return False
            else:
                # ANY inclusion pattern must match
                if patterns and not any(pattern.search(key) for pattern in patterns):
                    return False
            
            # Check exclusion patterns - if ANY exclusion pattern matches, reject the key
            if any(pattern.search(key) for pattern in exclusion_patterns):
                return False
            
            return True
        
        # Filter by specific log_type if provided
        if log_type:
            if log_type in domain_keys:
                keys_list = domain_keys[log_type]
                for key in keys_list:
                    if key_matches_criteria(key):
                        matching_keys.append(key)
        else:
            # Search in all log types
            for current_log_type, keys_list in domain_keys.items():
                for key in keys_list:
                    if key_matches_criteria(key):
                        matching_keys.append(key)
        
        return matching_keys