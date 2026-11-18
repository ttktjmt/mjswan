import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';
// Cut to the components the app renders: mantineCssPlugin in vite.config.ts.
import '@mantine/core/styles.css';

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
