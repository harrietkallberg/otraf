import React from 'react'
import { NavLink } from 'react-router-dom'

const Sidebar: React.FC = () => {
  const navItem = (to: string, label: string) => (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `block px-4 py-2 rounded hover:bg-gray-200 ${
          isActive ? 'bg-gray-300 font-semibold' : ''
        }`
      }
    >
      {label}
    </NavLink>
  )

  return (
    <div className="w-64 bg-gray-100 p-4 border-r h-full">
      <h2 className="text-xl font-bold mb-4">Transit Dashboard</h2>
      {navItem('/', 'Dashboard')}
      {navItem('/travel-times', 'Travel Times')}
      {navItem('/routes', 'Routes')}
      {navItem('/stops',  'Stops')}
      {navItem('/export-csv', 'Export CSV')}  {/* ← new */}
    </div>
  )
}

export default Sidebar
