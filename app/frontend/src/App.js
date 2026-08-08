import React from 'react';
import FileUpload from './FileUpload';
import './App.css';


const App = () => {
  return (
    <div className="App">
        <h1 className='title'>CliniDoc Predictor</h1>
        <h3>"For Clinical Document Classification"</h3>
        <FileUpload />

    </div>
  );
};

export default App;
