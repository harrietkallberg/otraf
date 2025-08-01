import React from 'react';
import { useAuth } from '../contexts/AuthContext';

const downloads = [
  {
    href: '/api/csv/travel_times',
    fileName: 'global_travel_times.csv',
    title: 'Global Travel Times',
    description: 'Aggregated mean & per-route travel times between every stop segment and time-of-day.',
  },
  {
    href: '/api/csv/underperforming_regulatory_stops',
    fileName: 'underperforming_regulatory_stops.csv',
    title: 'Underperforming Regulatory Stops',
    description: 'All regulatory stops whose overall on-time percentage fell below the performance threshold.',
  },
  {
    href: '/api/csv/mis_tracked_stops',
    fileName: 'mis_tracked_stops.csv',
    title: 'Mis-tracked Stops',
    description: 'Stops flagged for topology violations, with counts and max severity per stop.',
  },
];

const ExportCsvPage: React.FC = () => {
  const { user, session } = useAuth();

  const handleDownload = async (href: string, fileName: string) => {
    try {
      // Check if the session is valid
      if (!session || !session.access_token || !session.refresh_token || !user) {
        throw new Error('No valid session or tokens available');
      }

      const res = await fetch(href, {
        method: 'GET',
        headers: {
          'X-User-Id': user.id,  // Pass user ID
          'Authorization': `Bearer ${session.access_token}`,  // Pass access token
          'X-Refresh-Token': session.refresh_token,  // Pass refresh token in a custom header
        },
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

      // Create a blob from the response and set the proper UTF-8 encoding
      const blob = await res.blob();
      const utf8Blob = new Blob([await blob.text()], { type: 'text/csv; charset=utf-8' });

      // Create a URL for the blob and trigger the download
      const url = window.URL.createObjectURL(utf8Blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      a.remove();

      // Clean up the object URL after the download
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed', err);
      alert('Failed to download file. Please try again.');
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Export CSV</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {downloads.map((d) => (
          <div key={d.href} className="border p-4 rounded shadow-sm">
            <h2 className="font-semibold mb-2">{d.title}</h2>
            <p className="text-sm mb-4">{d.description}</p>
            <button
              onClick={() => handleDownload(d.href, d.fileName)}
              className="inline-block px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Download
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExportCsvPage;