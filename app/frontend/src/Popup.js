import React from 'react';
import './Popup.css';

const Popup = ({ message, closePopup }) => {
  return (
    <div className="popup-overlay">
      <div className="popup">
        <button className="close-btn" onClick={closePopup}>X</button>
        <h2 className='title-predict'>Label Comparison</h2>
        <p className="popup-content">
          {message}
        </p>
      </div>
    </div>
  );
};

export default Popup;
