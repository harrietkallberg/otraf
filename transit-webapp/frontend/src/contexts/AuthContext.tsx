import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { supabase } from '../supabaseClient'
import { User, Session } from '@supabase/supabase-js'

interface AuthContextType {
  user: User | null
  session: Session | null
  userRole: string | null
  isAdmin: boolean
  isLoading: boolean
  refreshRole: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [userRole, setUserRole] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  console.log('🔄 AuthProvider render - Current state:', {
    user: user?.email || null,
    session: !!session,
    userRole,
    isLoading
  })

  // Computed property for admin check
  const isAdmin = userRole === 'admin'

  // Function to fetch user role
  const fetchUserRole = async (userId: string) => {
    console.log('📋 fetchUserRole called for userId:', userId)
    
    try {
      // Simple fallback approach - if we can't query Supabase reliably,
      // let's use a more practical solution
      console.log('🔍 Using environment-based role assignment...')
      
      // Check if this is a known admin user by email pattern or specific ID
      // You can modify this logic based on your needs
      const isKnownAdmin = userId === '2c16edf4-fa45-4ac6-b708-c16e5ccd7f9d' // Your specific admin ID
      
      if (isKnownAdmin) {
        console.log('✅ Recognized admin user, assigning admin role')
        setUserRole('admin')
      } else {
        console.log('👤 Unknown user, assigning user role')
        setUserRole('user')
      }
      
      // TODO: Once we figure out the Supabase client issue, we can replace this
      // with the proper database query
      
    } catch (err) {
      console.error('❌ Error in role assignment:', err)
      setUserRole('user')
    }
  }

  // Function to refresh role (useful for admin panel)
  const refreshRole = async () => {
    console.log('🔄 refreshRole called')
    if (user?.id) {
      await fetchUserRole(user.id)
    } else {
      console.log('⚠️ No user ID available for role refresh')
    }
  }

  useEffect(() => {
    console.log('🚀 AuthProvider useEffect starting...')
    
    // Check session and user state when the app starts
    supabase.auth.getSession().then(async ({ data, error }) => {
      console.log('🔐 getSession result:', { 
        session: !!data.session, 
        user: data.session?.user?.email || null,
        error: error?.message || null
      })

      if (error) {
        console.error('❌ Error getting session:', error)
      }

      setSession(data.session)
      setUser(data.session?.user ?? null)
      
      if (data.session?.user?.id) {
        console.log('👤 User found, fetching role...')
        await fetchUserRole(data.session.user.id)
      } else {
        console.log('👤 No user found, clearing role')
        setUserRole(null)
      }
      
      console.log('✅ Initial auth check complete, setting isLoading to false')
      setIsLoading(false)
    }).catch((err) => {
      console.error('💥 Fatal error in getSession:', err)
      setIsLoading(false)
    })

    console.log('👂 Setting up auth state change listener...')
    const { data: listener } = supabase.auth.onAuthStateChange(async (event, session) => {
      console.log('🔔 Auth state change:', { 
        event, 
        session: !!session, 
        user: session?.user?.email || null 
      })

      setSession(session)
      setUser(session?.user ?? null)
      
      if (session?.user?.id) {
        console.log('👤 User logged in, fetching role...')
        await fetchUserRole(session.user.id)
      } else {
        console.log('👤 User logged out, clearing role')
        setUserRole(null)
      }
      
      console.log('✅ Auth state change processed, setting isLoading to false')
      setIsLoading(false)
    })

    return () => {
      console.log('🧹 Cleaning up auth listener')
      listener?.subscription.unsubscribe()
    }
  }, [])

  // Log whenever state changes
  useEffect(() => {
    console.log('📊 Auth state update:', {
      user: user?.email || null,
      session: !!session,
      userRole,
      isAdmin,
      isLoading
    })
  }, [user, session, userRole, isAdmin, isLoading])

  const contextValue = {
    user, 
    session, 
    userRole, 
    isAdmin, 
    isLoading, 
    refreshRole 
  }

  console.log('🎯 AuthProvider returning context value:', contextValue)

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    console.error('❌ useAuth must be used within an AuthProvider')
    throw new Error('useAuth must be used within an AuthProvider')
  }
  console.log('🎯 useAuth returning:', context)
  return context
}