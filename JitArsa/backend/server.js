const express = require('express');
const app = express();
const port = 3000;

app.use(express.json());

app.post('/ask-pha', async (req, res) => {
    try {
        const { question, history } = req.body;

        const response = await fetch('http://127.0.0.1:8000/ask-pha', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, history })
        });

        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error('Connection to Python failed:', error);
        res.status(500).json({ error: "น้องภาหลับอยู่จ้า ลองใหม่นะ" });
    }
});

app.listen(port, () => {
    console.log(`Node.js Proxy running at http://localhost:${port}`);
});