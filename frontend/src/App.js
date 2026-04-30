import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [eventLog, setEventLog] = useState([]);
  const [error, setError] = useState(null);

  // Fetch event log every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://127.0.0.1:5000/event_log")
        .then((res) => {
          if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
          }
          return res.json();
        })
        .then((data) => {
          setEventLog(data);
          setError(null);
        })
        .catch((err) => {
          console.error("Error fetching event log:", err);
          setError(err.message);
        });
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="App">
      <div className="camera-feed">
        <div className="camera-box">
          <img src="http://127.0.0.1:5000/video_feed" alt="Camera Feed" />
        </div>
      </div>
      <div className="event-log">
        <h3>Event Log</h3>
        <textarea
          readOnly
          value={eventLog.join("\n")}
          rows={10}
          cols={50}
        />
      </div>
      {error && <div className="error">Error: {error}</div>}
    </div>
  );
}

export default App;