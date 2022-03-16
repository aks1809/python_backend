import express from 'express';
import cors from 'cors';

const allowedDomains = ['http://192.168.1.10:3000'];

export default app => {
  app.use(express.json());
  app.use(
    cors({
      credentials: true,
      origin: (origin, callback) => {
        // bypass the requests with no origin (like curl requests, mobile apps, etc )
        if (!origin) return callback(null, true);

        if (allowedDomains.indexOf(origin) === -1) {
          const msg = `This site ${origin} does not have an access. Only specific domains are allowed to access it.`;
          return callback(new Error(msg), false);
        }
        return callback(null, true);
      }
    })
  );
};
