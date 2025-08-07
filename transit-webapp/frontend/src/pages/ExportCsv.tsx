import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext';

import { PageHeader } from '../components/shared'

const downloads = [
  {
    href: '/api/csv/travel_times',
    fileName: 'global_travel_times.csv',
    title: 'Global Travel Times',
    description: 'Aggregated mean & per-route travel times between every stop segment and time-of-day.',
    icon: (
      <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <circle cx="4" cy="12" r="3" fill="none" />
        <circle cx="20" cy="12" r="3" fill="none" />
        <line x1="7" y1="12" x2="17" y2="12" strokeLinecap="round" />
      </svg>
    ),
    iconBg: 'bg-indigo-100',
    category: 'Performance Data'
  },
  {
    href: '/api/csv/underperforming_regulatory_stops',
    fileName: 'underperforming_regulatory_stops.csv',
    title: 'Underperforming Regulatory Stops',
    description: 'All regulatory stops whose overall on-time percentage fell below the performance threshold.',
    icon: (
      <svg className="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
      </svg>
    ),
    iconBg: 'bg-amber-100',
    category: 'Regulatory Data'
  },
  {
    href: '/api/csv/mis_tracked_stops',
    fileName: 'mis_tracked_stops.csv',
    title: 'Mis-tracked Stops',
    description: 'Stops flagged for topology violations, with counts and max severity per stop.',
    icon: (
      <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    iconBg: 'bg-red-100',
    category: 'Data Quality'
  },
];

interface DownloadState {
  [key: string]: 'idle' | 'downloading' | 'success' | 'error';
}

const ExportCsvPage: React.FC = () => {
  const { user, session } = useAuth();
  const [downloadStates, setDownloadStates] = useState<DownloadState>({});
  const helpText =  'This page provides CSV exports of transit system data including travel times between stops, regulatory stops that are underperforming, and stops with data quality issues. All files use UTF-8 encoding and include headers for easy identification. Downloads are filtered based on your access permissions and may take a few moments for large datasets.'
  const handleDownload = async (href: string, fileName: string) => {
    try {
      // Check if the session is valid
      if (!session || !session.access_token || !session.refresh_token || !user) {
        throw new Error('No valid session or tokens available');
      }

      setDownloadStates(prev => ({ ...prev, [href]: 'downloading' }));

      const res = await fetch(href, {
        method: 'GET',
        headers: {
          'X-User-Id': user.id,
          'Authorization': `Bearer ${session.access_token}`,
          'X-Refresh-Token': session.refresh_token,
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
      
      setDownloadStates(prev => ({ ...prev, [href]: 'success' }));
      
      // Reset to idle after 2 seconds
      setTimeout(() => {
        setDownloadStates(prev => ({ ...prev, [href]: 'idle' }));
      }, 2000);

    } catch (err) {
      console.error('Download failed', err);
      setDownloadStates(prev => ({ ...prev, [href]: 'error' }));
      
      // Reset to idle after 3 seconds
      setTimeout(() => {
        setDownloadStates(prev => ({ ...prev, [href]: 'idle' }));
      }, 3000);
    }
  };

  const getButtonContent = (href: string, defaultText: string = 'Download') => {
    const state = downloadStates[href] || 'idle';
    switch (state) {
      case 'downloading':
        return (
          <span className="flex items-center space-x-2">
            <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            <span>Downloading...</span>
          </span>
        );
      case 'success':
        return (
          <span className="flex items-center space-x-2">
            <span>✓</span>
            <span>Downloaded</span>
          </span>
        );
      case 'error':
        return (
          <span className="flex items-center space-x-2">
            <span>✗</span>
            <span>Failed</span>
          </span>
        );
      default:
        return defaultText;
    }
  };

  const getButtonStyles = (href: string) => {
    const state = downloadStates[href] || 'idle';
    const baseStyles = "w-full px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center justify-center";
    
    switch (state) {
      case 'downloading':
        return `${baseStyles} bg-blue-200 text-blue-600 cursor-not-allowed`;
      case 'success':
        return `${baseStyles} bg-emerald-200 text-emerald-600`;
      case 'error':
        return `${baseStyles} bg-red-200 text-red-600 hover:bg-red-700`;
      default:
        return `${baseStyles} bg-blue-200 text-blue-600 hover:bg-blue-300 hover:shadow-md active:transform active:scale-95`;
    }
  };

  return (
    <div className="p-6 space-y-6">
      <PageHeader 
        title="Export CSV"
        helpText={helpText}
      />

      {/* Download Cards - All in one row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {downloads.map((download) => (
          <div
            key={download.href}
            className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow duration-200 flex flex-col"
          >
            {/* Header with icon */}
            <div className="flex items-center space-x-3 mb-4">
              <div className={`flex-shrink-0 w-10 h-10 ${download.iconBg} rounded-lg flex items-center justify-center`}>
                {download.icon}
              </div>
              <h3 className="text-lg font-semibold text-gray-900 leading-tight">
                {download.title}
              </h3>
            </div>

            {/* Description */}
            <p className="text-sm text-gray-600 mb-6 leading-relaxed flex-grow">
              {download.description}
            </p>

            {/* File info */}
            <div className="mb-4 p-3 bg-gray-50 rounded-md">
              <div className="text-xs text-gray-500 mb-1">File name:</div>
              <div className="text-sm font-mono text-gray-700">{download.fileName}</div>
            </div>

            {/* Download button */}
            <button
              onClick={() => handleDownload(download.href, download.fileName)}
              disabled={downloadStates[download.href] === 'downloading'}
              className={getButtonStyles(download.href)}
            >
              {getButtonContent(download.href)}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExportCsvPage;