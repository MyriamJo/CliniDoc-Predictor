# CliniDoc Predictor — Frontend

React.js interface for the CliniDoc Predictor demo. Lets a user upload a clinical document (`.txt`), preview its contents, and see the predicted category returned by the backend's Voting Classifier model.

## Structure

| File | Role |
|---|---|
| `src/App.js` | Page shell / title. |
| `src/FileUpload.js` | Core component: file picker, calls `POST /api/predict/` on the backend, shows a loading spinner, and displays the result in a popup. |
| `src/Popup.js` / `src/Popup.css` | Result popup showing actual vs. predicted label. |

## Setup

```
npm install
npm start
```

Runs on `http://localhost:3000`. Expects the backend (see `../backend`) running on `http://localhost:8000` — the API URL is currently hardcoded in `FileUpload.js` (`http://localhost:8000/api/predict/`).

## Demo

A recording of the app end-to-end (upload → content preview → predicted label) is available here: https://www.youtube.com/watch?v=OLQi3J8RsVM
