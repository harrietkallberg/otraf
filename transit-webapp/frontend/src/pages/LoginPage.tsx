// src/pages/LoginPage.tsx
import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../supabaseClient'
import { useAuth } from '../contexts/AuthContext'
import { useGlobalData } from '../contexts/GlobalDataContext'

const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isLoggingIn, setIsLoggingIn] = useState(false)
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0)
  const [isMessageVisible, setIsMessageVisible] = useState(true)
  const [loadingTimeout, setLoadingTimeout] = useState(false)
  const navigate = useNavigate()
  
  const { user, session, userRole, isLoading: authLoading, isUserRoleLoading } = useAuth()
  const globalData = useGlobalData()

  const loadingMessages = [
    "Spinning up your insights...",
    "Verifying authentication...",
    "Loading transit data...", 
    "Preparing analytics...",
    "Fetching route information...",
    "Setting up your dashboard...",
    "Almost ready..."
  ];

  // Fallback timeout to prevent infinite loading
  useEffect(() => {
    if (!isLoggingIn) return;

    const timeout = setTimeout(() => {
      console.log('⚠️ Loading timeout reached - checking what failed to load');
      setLoadingTimeout(true);
    }, 30000); // 30 second timeout

    return () => clearTimeout(timeout);
  }, [isLoggingIn]);

  // Rotate loading messages with fade transition
  useEffect(() => {
    if (!isLoggingIn) return;
    
    const interval = setInterval(() => {
      // Fade out current message
      setIsMessageVisible(false);
      
      // After fade out completes, change message and fade in
      setTimeout(() => {
        setCurrentMessageIndex(prev => (prev + 1) % loadingMessages.length);
        setIsMessageVisible(true);
      }, 250); // Half of the CSS transition duration
    }, 2500); // Slightly longer to account for transition time

    return () => clearInterval(interval);
  }, [isLoggingIn, loadingMessages.length]);

  // Check if fully loaded and navigate
  useEffect(() => {
    console.log('🔍 Login page loading state check:', {
      isLoggingIn,
      user: !!user,
      session: !!session,
      userRole,
      authLoading,
      isUserRoleLoading,
      globalData: !!globalData
    });

    if (isLoggingIn && user && session && !authLoading && !isUserRoleLoading && globalData) {
      console.log('✅ All loading complete, navigating to dashboard');
      // Small delay for better UX
      setTimeout(() => {
        navigate('/')
      }, 500);
    }
  }, [isLoggingIn, user, session, userRole, authLoading, isUserRoleLoading, globalData, navigate]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoggingIn(true)
    
    const { error } = await supabase.auth.signInWithPassword({ email, password })

    if (error) {
      setError(error.message)
      setIsLoggingIn(false)
    }
    // Don't navigate here - let the useEffect handle it when everything is loaded
  }

  // Show loading tile after login attempt
  if (isLoggingIn) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-white px-4 relative overflow-hidden">
        {/* Same background as original */}
        <svg 
          className="absolute inset-0 w-full h-full z-0" 
          viewBox="0 0 1200 800" 
          preserveAspectRatio="xMidYMid slice"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <radialGradient id="yellowRing" cx="50%" cy="50%" r="50%">
              <stop offset="70%" stopColor="transparent"/>
              <stop offset="70%" stopColor="#ffa70eff"/>
            </radialGradient>
            
            <radialGradient id="redRing" cx="50%" cy="50%" r="50%">
              <stop offset="70%" stopColor="#ffffffff"/>
              <stop offset="70%" stopColor="#b91c1c"/>
            </radialGradient>
          </defs>
          
          <ellipse
            cx="1600"
            cy="400"
            rx="900"
            ry="700"
            fill="url(#redRing)"
            opacity="1"
            transform="rotate(15 90 40) scale(0.9, 1.3)"
          />
          
          <ellipse
            cx="400"
            cy="400"
            rx="600"
            ry="250"
            fill="url(#yellowRing)"
            opacity="1"
            transform="rotate(15 30 60) scale(1.5, 1.4)"
          />
        </svg>

        {/* Loading Tile */}
        <div className="bg-white shadow-2xl rounded-2xl p-8 max-w-md w-full relative z-50 border-2 border-gray-200" style={{boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05)'}}>
          <h1 className="text-3xl font-bold text-center text-grey-900 mb-2">Transit Informer</h1>
          <p className="text-center text-grey-900 mb-8">Your one-stop hub for data-driven insights.</p>

          <div className="flex flex-col items-center space-y-6">
            {/* Spinner with rounded line caps */}
            <div className="relative">
              {/* Outer red spinner */}
              <svg className="w-16 h-16 animate-spin" style={{animationDuration: '3s'}}>
                <circle 
                  cx="32" 
                  cy="32" 
                  r="28" 
                  fill="none" 
                  stroke="#ffffff" 
                  strokeWidth="4"
                />
                <circle 
                  cx="32" 
                  cy="32" 
                  r="28" 
                  fill="none" 
                  stroke="#b91c1c" 
                  strokeWidth="4"
                  strokeDasharray="175"
                  strokeDashoffset="50"
                  strokeLinecap="round"
                />
              </svg>
              
              {/* Inner yellow spinner */}
              <svg 
                className="absolute inset-2 w-12 h-12 animate-spin" 
                style={{
                  animationDuration: '2s',
                  animationDirection: 'reverse'
                }}
              >
                <circle 
                  cx="24" 
                  cy="24" 
                  r="20" 
                  fill="none" 
                  stroke="#ffffff" 
                  strokeWidth="4"
                />
                <circle 
                  cx="24" 
                  cy="24" 
                  r="20" 
                  fill="none" 
                  stroke="#ffa70eff" 
                  strokeWidth="4"
                  strokeDasharray="125"
                  strokeDashoffset="50"
                  strokeLinecap="round"
                />
              </svg>
            </div>

            {/* Loading Message with smooth fade transitions */}
            <div className="text-center">
              <p 
                className={`text-lg font-medium text-gray-900 transition-opacity duration-500 ease-in-out ${
                  isMessageVisible ? 'opacity-100' : 'opacity-0'
                }`}
              >
                {loadingMessages[currentMessageIndex]}
              </p>
              <p className="text-sm text-gray-600 mt-2">
                Please wait while we prepare your experience
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Show normal login form
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-white px-4 relative overflow-hidden">
      {/* Background SVG */}
      <svg 
        className="absolute inset-0 w-full h-full z-0" 
        viewBox="0 0 1200 800" 
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <radialGradient id="yellowRing" cx="50%" cy="50%" r="50%">
            <stop offset="70%" stopColor="transparent"/>
            <stop offset="70%" stopColor="#ffa70eff"/>
          </radialGradient>
          
          <radialGradient id="redRing" cx="50%" cy="50%" r="50%">
            <stop offset="70%" stopColor="#ffffffff"/>
            <stop offset="70%" stopColor="#b91c1c"/>
          </radialGradient>
        </defs>
        
        <ellipse
          cx="1600"
          cy="400"
          rx="900"
          ry="700"
          fill="url(#redRing)"
          opacity="1"
          transform="rotate(15 90 40) scale(0.9, 1.3)"
        />
        
        <ellipse
          cx="400"
          cy="400"
          rx="600"
          ry="250"
          fill="url(#yellowRing)"
          opacity="1"
          transform="rotate(15 30 60) scale(1.5, 1.4)"
        />
      </svg>

      {/* Login Form */}
      <div className="bg-white shadow-2xl rounded-2xl p-8 max-w-md w-full relative z-50 border-2 border-gray-200" style={{boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05)'}}>
        <h1 className="text-3xl font-bold text-center text-grey-900 mb-2">Transit Analyzer</h1>
        <p className="text-center text-grey-900 mb-6">Your Guide to Valuable Insights.</p>

        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
              disabled={isLoggingIn}
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
              required
              disabled={isLoggingIn}
            />
          </div>

          {error && <p className="text-red-600 text-sm text-center">{error}</p>}
          
          {loadingTimeout && (
            <div className="text-yellow-600 text-sm text-center p-3 bg-yellow-50 rounded-lg">
              Loading is taking longer than expected. Please refresh the page or contact support if this continues.
            </div>
          )}

          <button
            type="submit"
            disabled={isLoggingIn}
            className="w-full py-2 text-white rounded-lg transition disabled:opacity-50"
            style={{backgroundColor: '#b91c1c'}}
            onMouseOver={(e) => !isLoggingIn && (e.currentTarget.style.backgroundColor = '#991b1b')}
            onMouseOut={(e) => !isLoggingIn && (e.currentTarget.style.backgroundColor = '#b91c1c')}
          >
            {isLoggingIn ? 'Logging In...' : 'Log In'}
          </button>
        </form>
      </div>
    </div>
  )
}

export default LoginPage