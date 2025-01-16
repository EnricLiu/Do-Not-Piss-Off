const app = require('./app');
const PORT = 55555;

app.listen(PORT, () => {
    console.log(`Server is up on PORT ${PORT}`);
});