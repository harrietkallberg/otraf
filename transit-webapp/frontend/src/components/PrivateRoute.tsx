// src/components/PrivateRoute.tsx
import { Navigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { ReactNode } from 'react'

interface Props {
  children: ReactNode
}

export default function PrivateRoute({ children }: Props) {
  const { user } = useAuth()
  return user ? children : <Navigate to="/login" replace />
}
