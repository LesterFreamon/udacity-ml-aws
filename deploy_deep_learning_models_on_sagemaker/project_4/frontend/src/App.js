// src/App.js
import React from 'react';
import Header from './components/Header/Header';
import ImageUpload from './components/ImageUpload/ImageUpload';
import ClassificationResult from './components/ClassificationResult/ClassificationResult';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <ImageUpload />
      <ClassificationResult />
    </div>
  );
}

export default App;