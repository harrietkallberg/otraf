import React, { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';

const Search: React.FC = () => {
  const [searchParams] = useSearchParams();
  const query = searchParams.get('q') || '';
  const [results, setResults] = useState<Record<string, any[]> | null>(null);

  useEffect(() => {
    if (query.trim()) {
      fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then((res) => res.json())
        .then((data) => setResults(data.matches));
    }
  }, [query]);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Search Results for "{query}"</h2>

      {!results && <p>Loading...</p>}

      {results && Object.entries(results).every(([_, list]) => list.length === 0) && (
        <p className="text-gray-600">No results found.</p>
      )}

      {results && Object.entries(results).map(([category, items]) =>
        items.length > 0 ? (
          <div key={category}>
            <h3 className="text-lg font-semibold capitalize mb-2">{category}</h3>
            <ul className="border rounded p-2 space-y-1 bg-white">
              {items.map((item, i) => (
                <li key={i}>
                  <Link
                    to={(() => {
                      if (category === 'stops') return `/stops/${item.stop_id}`;
                      if (category === 'routes') return `/routes/${item.route_id}`;
                      if (category === 'directions') return `/routes/${item.route_id}?direction=${item.direction_id}`;
                      return `/search?q=${encodeURIComponent(query)}`;
                    })()}
                    className="text-blue-600 hover:underline"
                  >
                    {(() => {
                      if (category === 'stops') return `${item.meta.stop_name} (${item.stop_id})`;
                      if (category === 'routes') return `${item.meta.route_short_name} - ${item.meta.route_long_name}`;
                      if (category === 'directions') return `${item.route_id} | Direction ${item.direction_id}`;
                      if (category === 'labels') return item;
                      if (category === 'time_types') return item;
                      if (category === 'regulatory_flags') return item.key;
                      return '';
                    })()}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ) : null
      )}
    </div>
  );
};

export default Search;
