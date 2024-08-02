const API_URL = process.env.REACT_APP_API_URL;

export const classifyImage = async (file) => {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_URL}/classify`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Classification failed');
  }

  return response.json();
};