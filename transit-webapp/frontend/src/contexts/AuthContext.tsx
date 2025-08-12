import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { supabase } from '../supabaseClient';
import { User, Session } from '@supabase/supabase-js';

interface AuthContextType {
  user: User | null;
  session: Session | null;
  userRole: string | null;
  isLoading: boolean;
  isUserRoleLoading: boolean; // Add this to the interface
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isUserRoleLoading, setIsUserRoleLoading] = useState(false);

  const fetchUserRole = async (userId: string, accessToken: string, refreshToken: string) => {
    setIsUserRoleLoading(true);
    console.log('👤 Starting fetchUserRole for userId:', userId);
    console.log('🔍 Access token exists:', !!accessToken);
    console.log('🔍 Refresh token exists:', !!refreshToken);

    try {
      console.log('🔗 About to query Supabase user_roles table...');
      
      // Add a timeout to prevent hanging indefinitely
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Supabase query timeout after 10 seconds')), 15000);
      });

      console.log('📤 Executing Supabase query...');
      
      // Race between the query and timeout
      const queryPromise = supabase
        .from('user_roles')
        .select('role')
        .eq('user_id', userId)
        .single();

      const { data, error } = await Promise.race([queryPromise, timeoutPromise]) as any;

      console.log('📡 Response from Supabase:', { data, error });

      if (error) {
        console.error('🚨 Supabase error details:', {
          message: error.message,
          code: error.code,
          hint: error.hint,
          details: error.details
        });
        throw error;
      }

      console.log('✔️ User role fetched successfully:', data);
      setUserRole(data?.role ?? 'user');
    } catch (err) {
      console.error('❌ Caught error in fetchUserRole:', err);
      console.error('❌ Error type:', typeof err);
      console.error('❌ Error constructor:', err?.constructor?.name);
      
      // Check if it's a timeout error
      if (err instanceof Error && err.message.includes('timeout')) {
        console.error('⏰ Query timed out - there may be a connection or RLS policy issue');
      }
      
      // Check if it's a network error
      if (err instanceof TypeError && err.message.includes('fetch')) {
        console.error('🌐 Network error detected');
      }
      
      setUserRole('user');
    } finally {
      console.log('🏁 fetchUserRole finally block reached');
      setIsUserRoleLoading(false);
    }
  };

  useEffect(() => {
    console.log('🚀 AuthProvider useEffect starting...');
    
    // Check session and user state when the app starts
    supabase.auth.getSession().then(async ({ data, error }) => {
      if (error) {
        console.error('Error getting session:', error);
        setIsLoading(false);
        return;
      }

      console.log('Session data:', data);

      setSession(data.session);
      setUser(data.session?.user ?? null);

      // Only fetch role if session is valid and user is authenticated
      if (data.session?.user?.id) {
        console.log('👤 User found, fetching role...');
        await fetchUserRole(data.session.user.id, data.session.access_token, data.session.refresh_token);
      } else {
        setUserRole(null);
        setIsUserRoleLoading(false);
      }
      
      // Set auth loading to false after everything is complete
      setIsLoading(false);
    }).catch((err) => {
      console.error('Error in getSession:', err);
      setIsLoading(false);
      setIsUserRoleLoading(false);
    });

    const { data: listener } = supabase.auth.onAuthStateChange(async (event, session) => {
      console.log('🔔 Auth state change:', { event, session });

      setSession(session);
      setUser(session?.user ?? null);

      if (session?.user?.id) {
        console.log('👤 User logged in, fetching role...');
        await fetchUserRole(session.user.id, session.access_token, session.refresh_token);
      } else {
        setUserRole(null);
        setIsUserRoleLoading(false);
      }

      setIsLoading(false);
    });

    return () => {
      listener?.subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    console.log('📊 Auth state update:', { user, session, userRole, isLoading, isUserRoleLoading });
  }, [user, session, userRole, isLoading, isUserRoleLoading]);

  const contextValue = {
    user,
    session,
    userRole,
    isLoading,
    isUserRoleLoading
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};