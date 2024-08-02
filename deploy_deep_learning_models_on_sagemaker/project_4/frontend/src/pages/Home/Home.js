import React from 'react';
import Header from '../../components/Header/Header';
import ImageUpload from '../../components/ImageUpload/ImageUpload';
import ClassificationResult from '../../components/ClassificationResult/ClassificationResult';
import Loading from '../../components/common/Loading/Loading';
import useClassification from '../../hooks/useClassification';
import './Home.css';

const Home = () => {
  const { result, loading, error, classify } = useClassification();

  return (
    <div className="home">
      <Header />
      <main>
        <ImageUpload onUpload={classify} />
        {loading && <Loading />}
        {error && <p className="error">{error}</p>}
        {result && <ClassificationResult breed={result.breed} confidence={result.confidence} />}
      </main>
    </div>
  );
};

export default Home;