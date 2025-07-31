import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom'; // To fetch the stop ID from the URL
import { useAuth } from '../contexts/AuthContext';

const StopLayout: React.FC = () => {
  const { stopId } = useParams<{ stopId: string }>(); // Extract stopId from URL
  const { user, session } = useAuth();
  const [stopData, setStopData] = useState<any>(null); // Store stop-specific data here
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (!user || !session?.access_token) return;

    const fetchStopData = async () => {
      setLoading(true);
      setError(null);

      const headers = {
        'X-User-Id': user.id,
        'Authorization': `Bearer ${session.access_token}`,
        'X-Refresh-Token': session.refresh_token, // Pass refresh token in a custom header
      };

      try {
        // Fetching stop-specific data from the user's folder
        const response = await fetch(`/api/stops/${stopId}`, { headers });
        const data = await response.json();
        setStopData(data);
      } catch (err) {
        setError('PAY ME MORE');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchStopData();
  }, [stopId, user, session]);

  if (loading) return <div>Loading stop details...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="px-6 py-4">
      <h2 className="text-2xl font-semibold mb-6">Stop: {stopData?.stop_name}</h2>
      
      {/* Stop Details */}
      <div>
        <h3 className="text-xl">Stop Details</h3>
        <p><strong>Stop ID:</strong> {stopData?.stop_id}</p>
        <p><strong>Location:</strong> {stopData?.location || 'N/A'}</p>
      </div>

      {/* Performance Data (If applicable) */}
      {stopData?.performance && (
        <div>
          <h3 className="text-xl">Performance</h3>
          <p><strong>Average Travel Time:</strong> {stopData.performance?.avg_time || 'N/A'} seconds</p>
          <p><strong>Sample Size:</strong> {stopData.performance?.sample_size || 0}</p>
        </div>
      )}

      {/* Additional Information */}
      <div>
        {/* You can add additional sections for other stop-related data */}
      </div>
    </div>
  );
};

export default StopLayout;
