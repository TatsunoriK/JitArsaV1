const express = require('express');
const { spawn } = require('child_process');
const app = express();
const port = 3000;

app.get('/run-python', (req, res) => {
    const pythonProcess = spawn('python', ['main.py']);

    let result = '';
    pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            res.send({ status: 'success', output: result.trim() });
        } else {
            res.status(500).send({ status: 'error', code });
        }
    });
});

app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});