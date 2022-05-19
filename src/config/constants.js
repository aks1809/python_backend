require('dotenv').config();

const config = {
  PORT: process.env.PORT || 9000,
  DEVIATION: process.env.DEVIATION,
  CAMERA_BACKEND: process.env.CAMERA_BACKEND,
  CAMERA_FRONTEND: process.env.CAMERA_FRONTEND
};
export default config;
