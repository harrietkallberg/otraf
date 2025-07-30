import { Navigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { ReactNode, ReactElement } from 'react'

interface Props {
  children: ReactNode
}

export default function PrivateRoute({ children }: Props): ReactElement {
  const { user, isLoading } = useAuth()  // Destructure isLoading from useAuth()

  // While checking authentication status, show a loading spinner or message
  if (isLoading) {
    return <div>Loading...</div>  // You can replace this with a loading spinner if needed
  }

  // If the user is authenticated, render the children (protected content)
  if (user) {
    return <>{children}</>
  } else {
    // If not authenticated, redirect to the login page
    return <Navigate to="/login" replace />
  }
}
