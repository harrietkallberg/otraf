// src/pages/ExportCsvPage.tsx
import React from 'react'

const downloads = [
  {
    href: '/api/csv/global_travel_times.csv',
    fileName: 'global_travel_times.csv',
    title: 'Global Travel Times',
    description:
      'Aggregated mean & per-route travel times between every stop segment and time-of-day.',
  },
  {
    href: '/api/csv/underperforming_regulatory_stops.csv',
    fileName: 'underperforming_regulatory_stops.csv',
    title: 'Underperforming Regulatory Stops',
    description:
      'All regulatory stops whose overall on-time percentage fell below the performance threshold.',
  },
  {
    href: '/api/csv/mis_tracked_stops.csv',
    fileName: 'mis_tracked_stops.csv',
    title: 'Mis-tracked Stops',
    description:
      'Stops flagged for topology violations, with counts and max severity per stop.',
  },
]

const ExportCsvPage: React.FC = () => (
  <div className="p-6">
    <h1 className="text-3xl font-bold mb-6">Export CSV</h1>
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
      {downloads.map(d => (
        <div key={d.href} className="border p-4 rounded shadow-sm">
          <h2 className="font-semibold mb-2">{d.title}</h2>
          <p className="text-sm mb-4">{d.description}</p>
          {/* **Must** be a plain <a> so the browser does a full GET (and invokes your proxy) */}
          <a
            href={d.href}
            download={d.fileName}
            className="inline-block px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Download
          </a>
        </div>
      ))}
    </div>
  </div>
)

export default ExportCsvPage
