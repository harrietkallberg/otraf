// src/components/Topbar.tsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface TopbarProps {
  title: string;
}

const Topbar: React.FC<TopbarProps> = ({ title }) => {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };

  return (
    <div className="flex items-center justify-between px-6 py-3 bg-white shadow border-b sticky top-0 z-50">
      <h1 className="text-xl font-semibold text-gray-800">{title}</h1>
      <form onSubmit={handleSearch} className="flex items-center space-x-2">
        <input
          type="text"
          placeholder="Search stops, routes, labels..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="px-3 py-1.5 border rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 w-72"
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-1.5 rounded-lg hover:bg-blue-700 transition"
        >
          Search
        </button>
      </form>
    </div>
  );
};

export default Topbar;
