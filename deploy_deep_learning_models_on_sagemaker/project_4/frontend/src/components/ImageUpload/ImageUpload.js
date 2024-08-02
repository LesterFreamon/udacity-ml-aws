import React, { useState } from 'react';
import './ImageUpload.css';
import Button from '../common/Button/Button';

const ImageUpload = ({ onUpload }) => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (file) onUpload(file);
  };

  return (
    <form onSubmit={handleSubmit} className="image-upload">
      <input type="file" onChange={handleFileChange} accept="image/*" />
      <Button type="submit" disabled={!file}>Classify</Button>
    </form>
  );
};

export default ImageUpload;




