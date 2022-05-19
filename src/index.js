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

const BASE_PATH = '/home/user/frinks/python_backend';

middlewaresConfig(app);

app.get('/', (req, res) => {
  return res.send('Home route is working');
});

app.get('/images', (req, res) => {
  try {
    const filepath = `${BASE_PATH}/images/${req.query.params}`;
    return res.sendFile(filepath);
  } catch (err) {
    console.log('images -------- error', err);
    return res.send('Error occured');
  }
});

app.get('/upload', async (req, res) => {
  try {
    // execute query on camera_backend
    const imageData = await axios.get(`${constants.CAMERA_BACKEND}/capture`);
    const base64Data = imageData.data.replace(/^data:image\/png;base64,/, '');
    fs.writeFile(`${BASE_PATH}/images/upload.bmp`, base64Data, 'base64', err => {
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
  } catch (err) {
    console.log('upload -------- error', err);
    res.send('Error occured');
  }
});

app.get('/analysis', (req, res) => {
  try {
    const pythonData = spawn('python3', [path.resolve(process.cwd(), 'scripts/main.py')]);
    console.log(pythonData);
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
  } catch (err) {
    console.log('analysis -------- error', err);
    res.send('Error occured');
  }
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
