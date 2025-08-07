// src/pages/LoginPage.tsx
import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../supabaseClient'

const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    const { error } = await supabase.auth.signInWithPassword({ email, password })

    if (error) {
      setError(error.message)
    } else {
      navigate('/')
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-white px-4 relative overflow-hidden">
      {/* Abstract Organic Curves Background */}
      <svg 
        className="absolute inset-0 w-full h-full z-0" 
        viewBox="0 0 1200 800" 
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
      ></svg>
      {/* Abstract Organic Curves Background */}
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
        
        {/* Massive red ring - center right */}
        <ellipse
          cx="1600"
          cy="400"
          rx="900"
          ry="700"
          fill="url(#redRing)"
          opacity="1"
          transform="rotate(15 90 40) scale(0.9, 1.3)"
        />
        
        {/* Large yellow ring - bottom left */}
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

      {/* Login Form - Elevated above background */}
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
            />
          </div>

          {error && <p className="text-red-600 text-sm text-center">{error}</p>}

          <button
            type="submit"
            className="w-full py-2 text-white rounded-lg transition"
            style={{backgroundColor: '#b91c1c'}}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#991b1b'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#b91c1c'}
          >
            Log In
          </button>
        </form>
      </div>
    </div>
  )
}

export default LoginPage