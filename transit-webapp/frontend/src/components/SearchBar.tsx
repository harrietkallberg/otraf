// src/components/SearchBar.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const SearchBar: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any>(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const delayDebounce = setTimeout(() => {
      if (query.trim()) {
        fetch(`/api/search?q=${encodeURIComponent(query)}`)
          .then(res => res.json())
          .then(data => {
            setResults(data.matches);
            setShowDropdown(true);
          });
      } else {
        setResults(null);
        setShowDropdown(false);
      }
    }, 300);

    return () => clearTimeout(delayDebounce);
  }, [query]);

  const handleSelect = (type: string, item: any) => {
    setQuery('');
    setShowDropdown(false);
    if (type === 'stops') navigate(`/stops/${item.stop_id}`);
    else if (type === 'routes') navigate(`/routes/${item.route_id}`);
    else if (type === 'directions') navigate(`/routes/${item.route_id}?direction=${item.direction_id}`);
    else if (type === 'labels') navigate(`/violations?q=${item}`);
    else if (type === 'time_types') navigate(`/analytics?q=${item}`);
    else if (type === 'regulatory_flags') navigate(`/analytics?q=${item.key}`);
  };

  return (
    <div className="relative w-full max-w-3xl mx-auto mt-4">
      <input
        className="w-full px-4 py-2 rounded border border-gray-300"
        placeholder="Search stops, routes, directions, labels..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {showDropdown && results && (
        <div className="absolute w-full bg-white border mt-1 rounded shadow z-50 max-h-96 overflow-y-auto">
          {Object.entries(results).map(([type, items]: any) => (
            items.length > 0 && (
              <div key={type}>
                <div className="bg-gray-100 px-4 py-1 text-xs uppercase text-gray-500">{type}</div>
                {items.map((item: any, i: number) => (
                  <div
                    key={i}
                    onClick={() => handleSelect(type, item)}
                    className="px-4 py-2 hover:bg-blue-100 cursor-pointer"
                  >
                    {type === 'stops' && `${item.meta.stop_name} (${item.stop_id})`}
                    {type === 'routes' && `${item.meta.route_short_name} - ${item.meta.route_long_name}`}
                    {type === 'directions' && `${item.route_id} | Direction ${item.direction_id}`}
                    {type === 'labels' && item}
                    {type === 'time_types' && item}
                    {type === 'regulatory_flags' && item.key}
                  </div>
                ))}
              </div>
            )
          ))}
        </div>
      )}
    </div>
  );
};

export default SearchBar;
