const path = require('path');
const express = require('express');
const router = require('./route');
const app = express();

// 静态资源
app.use(express.static(path.join(__dirname, '../frontend')));
app.use('/api', router);

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
})
module.exports = app;
