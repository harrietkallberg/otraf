import React from 'react'
import { NavLink } from 'react-router-dom'

const Sidebar: React.FC = () => {
  const navItem = (to: string, label: string, icon: React.ReactNode, iconColor?: string) => (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center space-x-3 px-4 py-2 rounded hover:bg-gray-200 transition-colors ${
          isActive ? 'bg-gray-300 font-semibold' : ''
        }`
      }
    >
      <span className={iconColor || 'text-gray-600'}>
        {icon}
      </span>
      <span>{label}</span>
    </NavLink>
  )

  return (
    <div className="w-64 bg-gray-100 p-4 border-r h-full">
      <h2 className="text-xl font-bold mb-4">Transit Analyzer</h2>
      <div className="space-y-1">
        {navItem('/', 'Dashboard', 
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
        )}
        
        {navItem('/routes', 'Routes', 
          <div className="w-5 h-5 border-2 border-sky-600 rounded flex items-center justify-center">
            <span className="text-sky-600 text-[8px] font-extrabold leading-none">123</span>
          </div>, 
          ''
        )}
        
        {navItem('/stops', 'Stops', 
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>, 
          'text-orange-600'
        )}
        
        {navItem('/explore-logs', 'Explore Logs', 
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>, 
          'text-amber-600'
        )}
        
        {navItem('/travel-times', 'Travel Times', 
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <circle cx="4" cy="12" r="3" />
            <circle cx="20" cy="12" r="3" />
            <line x1="7" y1="12" x2="17" y2="12" strokeLinecap="round" />
          </svg>, 
          'text-indigo-600'
        )}
        
        {navItem('/export-csv', 'Export CSV', 
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        )}
      </div>
    </div>
  )
}

export default Sidebar