import path from 'path';
import axios from 'axios';
import express from 'express';
import { createServer } from 'http';
import { spawn } from 'child_process';
import fs from 'fs';

import middlewaresConfig from './config/middleware';
import constants from './config/constants';

const app = express();
const httpServer = createServer(app);

middlewaresConfig(app);

app.get('/images', (req, res) => {
  const filepath = `/python_backend/images/${req.query.params}`;
  return res.sendFile(filepath);
});

app.get('/upload', async (req, res) => {
  // execute query on camera_backend
  const imageData = await axios.get(`${constants.CAMERA_BACKEND}/capture`);
  const base64Data = imageData.data.replace(/^data:image\/png;base64,/, '');
  fs.writeFile('/python_backend/images/upload.bmp', base64Data, 'base64', err => {
    if (err) {
      console.log(err);
    }
  });
  res.status(200).json({
    success: true,
    status: 'Uploaded',
    data: {
      name: 'upload.bmp'
    }
  });
});

app.get('/analysis', (req, res) => {
  const pythonData = spawn('python3', [path.resolve(process.cwd(), 'scripts/main.py')]);
  pythonData.stdout.on('data', data => {
    res.status(200).json({
      success: true,
      status: 'Result Found',
      data: {
        analysis: JSON.parse(data),
        dividend: constants.DEVIATION
      }
    });
  });
});

if (!module.parent) {
  httpServer.listen(constants.PORT, err => {
    if (err) {
      console.log('Cannot run!');
    } else {
      console.log(`API server listening on port: ${constants.PORT}`);
    }
  });
}

export default app;
