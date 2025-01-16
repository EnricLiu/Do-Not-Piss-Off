// 获取按钮和输入框元素
const heartMean = document.getElementById('heart_mean');
const heartFreq = document.getElementById('heart_freq');
const heartVary = document.getElementById('heart_vary');
const breathMean = document.getElementById('breath_mean');
const breathFreq = document.getElementById('breath_freq');
const breathVary = document.getElementById('breath_vary');

const applyHeartBtn = document.getElementById('override_heart');
const applyBreathBtn = document.getElementById('override_breath');
const cancelHeartBtn = document.getElementById('cancel_heart');
const cancelBreathBtn = document.getElementById('cancel_breath');
const resetHeartBtn = document.getElementById('reset_heart');
const resetBreathBtn = document.getElementById('reset_breath');
const emoButtons = document.querySelectorAll('.emo-btn');

const pageUrl = document.URL;

const resetState = {
    heart: {
        mean: 75,
        range: 10,
        freq: 5
    },
    breath: {
        mean: 17,
        range: 5,
        freq: 1
    },
}

const overrideState = {
    heart: {
        is_override: false,
        mean: null,
        range: null
    },
    breath: {
        is_override: false,
        mean: null,
        range: null
    },
    emotion: {
        is_override: false,
        target: null
    }
}

const updateState = () => {
    fetch(`${pageUrl}api/override`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(overrideState),
    }).then(data => console.log("Success:", data))
      .catch(err => console.error("Error:", err));
}

emoButtons.forEach(button => {
    button.addEventListener('click', () => {
        if(overrideState.emotion.is_override) {
            if(overrideState.emotion.target === button.id) {
                overrideState.emotion.is_override = false;
                overrideState.emotion.target = null;
                button.innerHTML = button.innerHTML.slice(1);
            } else {
                const last_button = document.getElementById(overrideState.emotion.target);
                last_button.innerHTML = last_button.innerHTML.slice(1);
                overrideState.emotion.target = button.id;
                button.innerHTML = `@${button.innerHTML}`;
            }
        }
        else {
            overrideState.emotion.is_override = true;
            overrideState.emotion.target = button.id;
            button.innerHTML = `@${button.innerHTML}`;
        }
        updateState()
    });
});

applyHeartBtn.addEventListener('click', () => {
    if(!overrideState.heart.is_override) {
        applyHeartBtn.innerHTML = `@${applyHeartBtn.innerHTML}`;
    }
    overrideState.heart.is_override = true;
    overrideState.heart.mean  = heartMean.value;
    overrideState.heart.range = heartVary.value;
    updateState();
});

applyBreathBtn.addEventListener('click', () => {
    if(!overrideState.breath.is_override) {
        applyBreathBtn.innerHTML = `@${applyBreathBtn.innerHTML}`;
    }
    overrideState.breath.is_override = true;
    overrideState.breath.mean = breathMean.value;
    overrideState.breath.range = breathVary.value;
    updateState();
});

cancelHeartBtn.addEventListener('click', () => {
    if(overrideState.heart.is_override) {
        applyHeartBtn.innerHTML = applyHeartBtn.innerHTML.slice(1);
    }
    overrideState.heart.is_override = false;
    overrideState.heart.mean = null;
    overrideState.heart.range = null;
    updateState();
});

cancelBreathBtn.addEventListener('click', () => {
    if(overrideState.breath.is_override) {
        applyBreathBtn.innerHTML = applyBreathBtn.innerHTML.slice(1);
    }
    overrideState.breath.is_override = false;
    overrideState.breath.mean = null;
    overrideState.breath.range = null;
    updateState();
});

resetHeartBtn.addEventListener('click', () => {
    if(overrideState.heart.is_override) {
        applyHeartBtn.innerHTML = applyHeartBtn.innerHTML.slice(1);
    }
    overrideState.heart.is_override = false;
    overrideState.heart.mean = null;
    overrideState.heart.range = null;
    heartMean.value = resetState.heart.mean;
    heartVary.value = resetState.heart.range;
    updateState();
});

resetBreathBtn.addEventListener('click', () => {
    if(overrideState.breath.is_override) {
        applyBreathBtn.innerHTML = applyBreathBtn.innerHTML.slice(1);
    }
    overrideState.breath.is_override = false;
    overrideState.breath.mean = null;
    overrideState.breath.range = null;
    breathMean.value = resetState.breath.mean;
    breathVary.value = resetState.breath.range;
    updateState();
});