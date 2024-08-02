// src/components/ClassificationResult.js
import React from 'react';

const ClassificationResult = ({ breed, confidence }) => {
  return (
    <div>
      <h2>Classification Result:</h2>
      {breed ? (
        <p>
          Breed: {breed} (Confidence: {confidence.toFixed(2)}%)
        </p>
      ) : (
        <p>No classification result yet.</p>
      )}
    </div>
  );
};

export default ClassificationResult;