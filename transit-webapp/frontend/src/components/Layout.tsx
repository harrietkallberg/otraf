import React from 'react'
import Sidebar from './Sidebar'
import Topbar  from './Topbar'

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="flex h-screen">
    <Sidebar />
    <div className="flex-1 flex flex-col">
      <Topbar title="Transit Dashboard" />
      <main className="p-4 overflow-auto">{children}</main>
    </div>
  </div>
)

export default Layout
