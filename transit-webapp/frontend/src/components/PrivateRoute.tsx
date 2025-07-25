import { Navigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { ReactNode, ReactElement } from 'react'

interface Props {
  children: ReactNode
}

export default function PrivateRoute({ children }: Props): ReactElement {
  const { user } = useAuth()

  if (user) {
    return <>{children}</>
  } else {
    return <Navigate to="/login" replace />
  }
}
