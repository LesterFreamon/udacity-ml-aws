import { useState } from 'react';
import { classifyImage } from '../services/api';

const useClassification = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const classify = async (file) => {
    setLoading(true);
    setError(null);
    try {
      const data = await classifyImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return { result, loading, error, classify };
};

export default useClassification;