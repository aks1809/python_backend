require('dotenv').config();

const config = {
  PORT: process.env.PORT || 9000,
  DEVIATION: process.env.DEVIATION,
  CAMERA_BACKEND_BASE_PATH: process.env.CAMERA_BACKEND_BASE_PATH
};
export default config;
