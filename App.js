import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState("");

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setImageUrl(URL.createObjectURL(file));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post('http://localhost:5000/recognize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResult(response.data.animal);
    } catch (error) {
      console.error('Error recognizing animal:', error);
    }
  };

  return (
    <div className="App">
      <h1>Animal Recognition Application</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleImageUpload} />
        <button type="submit">Recognize Animal</button>
      </form>
      {imageUrl && (
        <div>
          <h2>Uploaded Image:</h2>
          <img src={imageUrl} alt="Uploaded Animal" style={{ maxWidth: '300px', maxHeight: '300px' }} />
        </div>
      )}
      {result && <h2>Recognized Animal: {result}</h2>}
    </div>
  );
}

export default App;
