// src/components/shared/AccessControl.tsx
import React from 'react'
import { useAuth } from '../../contexts/AuthContext'

interface AccessControlProps {
  children: React.ReactNode
  requireAdmin?: boolean
  fallback?: React.ReactNode
  showUpgradeMessage?: boolean
}

interface RestrictedContentOverlayProps {
  message?: string
  showUpgradePrompt?: boolean
}

const RestrictedContentOverlay: React.FC<RestrictedContentOverlayProps> = ({ 
  showUpgradePrompt = true 
}) => {
  return (
    <div className="absolute inset-x-0 top-16 z-20 flex flex-col items-center justify-start p-6 text-center">
      <div className="bg-white rounded-xl shadow-2xl p-8 max-w-md border border-gray-300 transform scale-105">
        {/* Lock Icon */}
        <div className="w-20 h-20 bg-blue-50 rounded-full flex border border-blue-200 items-center justify-center mx-auto mb-6 shadow-inner">
          <svg className="w-10 h-10 text-blue-800" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
        </div>
        
        <h3 className="text-xl font-bold text-blue-800 mb-3">
          Want the full picture?
        </h3>
        
        {showUpgradePrompt && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-3">
            <p className="text-sm text-blue-800 font-medium">
              Contact Harriet to unlock deep insights about your transit system.
            </p>
            <a
              href="mailto:harriet.kallberg02@gmail.com?subject=I%20Want%20the%20Full%20Picture%20-%20Transit%20Analyzer&body=Hi%20Harriet,%0A%0AI%20would%20like%20to%20unlock%20Premium%20Access%20to%20the%20Transit%20Analyzer%20system.%0A%0AThank%20you!"
              className="inline-flex items-center justify-center w-full px-4 py-2 bg-blue-800 text-white text-sm font-bold rounded-lg hover:bg-blue-500 transition-colors duration-200"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              Unlock Premium Access
            </a>
          </div>
        )}
      </div>
    </div>
  )
}


const AccessControl: React.FC<AccessControlProps> = ({ 
  children, 
  requireAdmin = false, 
  fallback,
  showUpgradeMessage = true 
}) => {
  const { userRole, isLoading } = useAuth()

  // While loading, show loading state
  if (isLoading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-8 bg-gray-200 rounded w-1/4"></div>
        <div className="h-32 bg-gray-200 rounded"></div>
      </div>
    )
  }

  // If admin access is required and user is not admin
  if (requireAdmin && userRole!=='admin') {
    if (fallback) {
      return <>{fallback}</>
    }

    return (
      <div className="relative min-h-[300px]">
        {/* Apply blur effect directly to content */}
        <div 
          className="pointer-events-none select-none transition-all duration-300"
          style={{
            filter: 'blur(5px) grayscale(70%)',
            opacity: 0.3
          }}
        >
          {children}
        </div>
        <RestrictedContentOverlay 
          showUpgradePrompt={showUpgradeMessage}
        />
      </div>
    )
  }

  // User has access, render normally
  return <>{children}</>
}

export { AccessControl, RestrictedContentOverlay }