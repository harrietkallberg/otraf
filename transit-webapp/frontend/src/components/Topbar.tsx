// src/components/Topbar.tsx
import React from 'react'

interface TopbarProps {
  title: string
  children?: React.ReactNode  // ✅ ALLOW CHILDREN HERE
}

const Topbar: React.FC<TopbarProps> = ({ title, children }) => {

  return (
    <div className="flex items-center justify-between px-6 py-3 bg-white shadow border-b sticky top-0 z-50">
      <h1 className="text-xl font-semibold text-gray-800">{title}</h1>
      {children && (
        <div className="ml-4">
          {children} {/* ✅ Renders logout button passed from Layout */}
        </div>
      )}
    </div>
  )
}

export default Topbar
