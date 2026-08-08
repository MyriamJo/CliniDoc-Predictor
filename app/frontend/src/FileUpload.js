import React, { useState } from 'react';
import axios from 'axios';
import Popup from './Popup';
import './Popup.css';
import { Oval } from 'react-loader-spinner';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [fileContent, setFileContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [documentLabel, setDocumentLabel] = useState('');
  const [error, setError] = useState(null);
  const [showPopup, setShowPopup] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    const transformedName = transformFileName(selectedFile.name);
    const newFile = new File([selectedFile], transformedName.displayName, { type: selectedFile.type });

    setFile(newFile);
    setFileName(transformedName.displayName);
    setDocumentLabel(transformedName.documentType);
    setFileContent('');
    setPrediction(null);
    setError(null);
  };

  const transformFileName = (name) => {
    const parts = name.split('_');
    const documentNumber = parts[0];
    const documentType = parts[1].split('.')[0];
    const extension = name.split('.').pop();
    const displayName = `Document_${documentNumber}.${extension}`;
    return { displayName, documentType };
  };

  const readFileContent = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        resolve(event.target.result);
      };
      reader.onerror = (error) => {
        reject(error);
      };
      reader.readAsText(file);
    });
  };

  const handleUploadClick = async () => {
    if (file) {
      setFileContent(await readFileContent(file));
    }
  };

  const handleFileUpload = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append('document', file);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/api/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log("Full response:", response); 
      if (response.data && response.data.prediction) {
        setPrediction(response.data.prediction);
        setShowPopup(true); 
      } else {
        setError("No prediction returned from the server.");
      }
    } catch (err) {
      console.error("Error occurred while uploading the file:", err);
      setError("An error occurred while uploading the file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form>
        <div className="file-input-container">
          <input
            type="file"
            id="file-input"
            className="upload-button"
            onChange={handleFileChange}
            style={{ display: 'none' }}  
          />
          <label htmlFor="file-input" className="button custom-file-input">
            Choose File
          </label>
            <span><b style={{color:'#b22222'}}>Selected File: </b></span>
            <span style={{width:'200px',height: '20px',color:'b2b2b2',marginLeft: '10px', padding: '6px', border: '2px solid #ccc', borderRadius: '4px' }}>{fileName ? `${fileName}` : ''}</span>
        </div>
        <button className="button upload-button" type="button" onClick={handleUploadClick}>Upload File</button>
        <button className="button classify-button" type="button" onClick={handleFileUpload}>Classify Document</button>
      </form>
      {loading && (
        <div className="loader-container">
          <Oval
            height={40}
            width={40}
            color="#b22222"
            visible={true}
            ariaLabel='oval-loading'
            secondaryColor="#ffffff"
            strokeWidth={2}
            strokeWidthSecondary={2}
          />
        </div>
      )}
      {showPopup && (
        <Popup
          message={`Actual Label: ${documentLabel}\n\nPredicted Label: ${prediction}`}
          closePopup={() => setShowPopup(false)}
        />
      )}
      {fileContent && (
        <div>
          <h3>Document Content:</h3>
          <textarea rows="35" cols="100" readOnly value={fileContent}></textarea>
        </div>
      )}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
};

export default FileUpload;