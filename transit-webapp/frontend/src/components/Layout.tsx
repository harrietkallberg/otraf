import React from 'react'
import Sidebar from './Sidebar'
import Topbar from './Topbar'
import { Outlet } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { supabase } from '../supabaseClient'

const Layout: React.FC = () => {
  const { user } = useAuth()
  console.log('Layout rendering, user:', !!user) // Add this line

  const handleLogout = async () => {
    await supabase.auth.signOut()
    window.location.href = '/login'
  }

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Topbar title="Transit Dashboard">
          {user && (
            <button
              onClick={handleLogout}
              className="ml-auto px-4 py-1 bg-red-600 text-white rounded hover:bg-red-700"
            >
              Logout
            </button>
          )}
        </Topbar>
        <main className="p-4 overflow-auto">
          <Outlet /> {/* ✅ Route children render here */}
        </main>
      </div>
    </div>
  )
}

export default Layout
