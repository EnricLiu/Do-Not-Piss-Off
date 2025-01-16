const fs = require('fs');

const express = require('express');
const { devNull } = require('os');
const router = express();
const verifyToken = (req, res, next) => {
    // const token = req.headers['authorization'];
    // if (!token) return res.status(401).send('Unauthorized');
    // if (!router.allValidTokens) return res.status(502).send('Bad Gateway');
    // if (!router.allValidTokens.includes(token)) {
    //     return res.status(401).send('Unauthorized');
    // }

    next();
}

router.updateValidTokens = function () {
    this.allValidTokens = fs.readFileSync('./valid_tokens.txt').toString().split('\n').map(t => { return t.trim() });
}.bind(router);

router.updateValidTokens();
router.use(verifyToken);
router.use(express.json());

let game_info = {
    type: "none"
};
const update_game_info = (new_info) => {
    if(!new_info?.type) throw new Error("Invalid request");
    switch (new_info["type"]) {
        case "none":
            game_info = {
                type: "none"
            }
            break;
        case "music":
            if (!new_info["name"]) throw new Error("Invalid request");
            game_info = {
                type:       "music",
                name:       new_info["name"],
                subtitle:   new_info["subtitle"] ?? null,
            }
            break;
        case "game":
            if (!new_info["name"]) throw new Error("Invalid request");
            if (!new_info["start_at"]) throw new Error("Invalid request");
            game_info = {
                type:       "game",
                name:       new_info["name"],
                subtitle:   new_info["subtitle"] ?? null,
                start_at:   new_info["start_at"],
            }
            break;
        default:
            throw new Error("Invalid request");
    }
}

let currEmotion = null;

const emotionList = ["happy", "sad", "angry", "disgust", "fear", "neutral"]
const update_emotion = (new_emotion) => {
    if(!new_emotion?.emotion) throw new Error("Invalid request");
    if(!emotionList.includes(new_emotion["emotion"])) throw new Error("Invalid request");

    currEmotion = new_emotion["emotion"];
}

router.get('/game_info', (req, res) => {
    res.json(game_info);
});

router.post('/game_info', (req, res) => {
    const body = req.body;
    try {
        update_game_info(body);
    } catch (e) {
        switch (e.message) {
            case "Invalid request": return res.status(400).send("Invalid request");
            default:                return res.status(500).send("Internal server error");
        }
    }
    res.send("ok");
});

router.get('/emotion', (req, res) => {
    res.json(currEmotion);
});

router.post('/emotion', (req, res) => {
    const body = req.body;
    try {
        update_emotion(body)
    } catch (e) {
        switch (e.message) {
            case "Invalid request": return res.status(400).send("Invalid request");
            default: return res.status(500).send("Internal server error");
        }
    }
    res.send("ok");
}); 

let gOverrideState = {
    "heart": {
        "is_override": false,
        "mean": null,
        "range": null
    },
    "breath": {
        "is_override": false,
        "mean": null,
        "range": null
    },
    "emotion": {
        "is_override": false,
        "target": null
    }
}

router.get('/override', (req, res) => {
    res.json(gOverrideState)
})

router.post('/override', (req, res) => {
    const body = req.body;
    gOverrideState = body;
    console.log(gOverrideState)
    res.send("ok");
})

module.exports = router;