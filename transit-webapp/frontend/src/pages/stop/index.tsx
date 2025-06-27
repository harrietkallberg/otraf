// src/pages/StopLayout.tsx
import React, {useEffect, useState} from 'react'
import {useParams, NavLink, Routes, Route, Navigate} from 'react-router-dom'
import StopOverview    from './Overview'
import StopPerformance from './Performance'
import StopViolations  from './Violations'

export default function StopLayout(){
  const { sid } = useParams<{sid:string}>()
  const [stop, setStop] = useState<any>(null)
  useEffect(()=>{
    fetch(`/api/stops/${sid}`).then(r=>r.json()).then(setStop)
  },[sid])
  if (!stop) return <div>Loading…</div>

  return (
    <>
      <h1 className="text-2xl font-bold mb-2">{stop.meta.stop_name}</h1>
      <nav className="flex space-x-4 mb-4 border-b">
        {['overview','performance','violations'].map(tab=>(
          <NavLink
            key={tab}
            to={tab}
            className={({isActive})=>isActive?'pb-1 border-b-2':'pb-1 hover:border-b'}
          >{tab.charAt(0).toUpperCase()+tab.slice(1)}</NavLink>
        ))}
      </nav>

      <Routes>
        <Route path="/"            element={<Navigate to="overview" replace />} />
        <Route path="overview"     element={<StopOverview data={stop}/>} />
        <Route path="performance"  element={<StopPerformance data={stop}/>} />
        <Route path="violations"   element={<StopViolations  data={stop}/>} />
      </Routes>
    </>
  )
}
