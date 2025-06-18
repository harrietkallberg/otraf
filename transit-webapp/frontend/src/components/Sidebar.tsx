// src/components/Sidebar.tsx
import React from 'react';
import { NavLink } from 'react-router-dom';

const Sidebar: React.FC = () => {
  const navItem = (to: string, label: string) => (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `block px-4 py-2 rounded hover:bg-gray-200 ${isActive ? 'bg-gray-300 font-semibold' : ''}`
      }
    >
      {label}
    </NavLink>
  );

  return (
    <div className="w-64 bg-gray-100 p-4 border-r h-full">
      <h2 className="text-xl font-bold mb-4">Transit Dashboard</h2>
      {navItem('/dashboard', 'Dashboard')}
      {navItem('/search', 'Search')}
      {navItem('/routes/overview', 'Routes')}
      {navItem('/stops/overview', 'Stops')}
      {navItem('/violations', 'Violations')}
      {navItem('/analytics', 'Analytics')}
    </div>
  );
};

export default Sidebar;