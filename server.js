/*
 * Copyright © 2018 Intel Corporation. All Rights Reserved.
 */
'use strict';

const express = require('express');
const fs = require('fs');
const https = require('https');
const app = express();

// Directory 'public' for static files
app.use(express.static(__dirname + '/', {
  setHeaders: (res) => {
    res.set('Cross-Origin-Opener-Policy', 'same-origin');
    res.set('Cross-Origin-Embedder-Policy', 'require-corp');
  }
}));
app.use(express.json());
app.use(express.urlencoded({
  extended: true
}));

// app.listen("8083");
// console.log('http://127.0.0.1:8083/semantic_segmentation/')

// Start HTTPS server
https.createServer({
    cert: fs.readFileSync('cert.pem'),
    key: fs.readFileSync('key.pem'),
}, app).listen("8088");
console.log('https://127.0.0.1:8088/semantic_segmentation/')