const pageUrl = document.URL;

function updateTime() {
    const now = new Date();
    const timeElement = document.querySelector('.time');
    const dateElement = document.querySelector('.date');

    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const seconds = now.getSeconds().toString().padStart(2, '0');
    const timeString = `${hours}:${minutes}:${seconds}`;

    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const dateString = now.toLocaleDateString('zh-CN', options);

    timeElement.textContent = timeString;
    dateElement.textContent = dateString;
}

const EmoMap = {
    "happy": {
        emoName: "😄",
        desc: "",
    },
    "sad": {
        emoName: "🙁",
        desc: "",
    },
    "angry": {
        emoName: "😠",
        desc: "",
    },
    "suprised": {
        emoName: "🤩",
        desc: "",
    },
    "disgust": {
        emoName: "🤮",
        desc: "",
    },
    "fear": {
        emoName: "😨",
        desc: "",
    },
    "neutral": {
        emoName: "🙂",
        desc: "",
    },
}

// 模拟数据更新 (实际应用中需要从真实数据源获取)
function updateData() {
    const heartRateElement = document.querySelector('.heart-rate .value');
    const breathRateElement = document.querySelector('.breath-rate .value');
    const emoNameElement = document.querySelector('.emo-info .emo-name');
    const emoDescElement = document.querySelector('.emo-info .emo-desc');

    const randHeartRate = Math.floor(Math.random() * 20) + 60; // 60-80 bpm
    const randBreathRate = Math.floor(Math.random() * 4) + 14;  // 14-18 breaths/min

    const url = `${pageUrl}/data`;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            console.log(`fetch data: ${JSON.stringify(data)}`);
            heartRateElement.textContent = data.heart ?? randHeartRate;
            breathRateElement.textContent = data.breath ?? randBreathRate;

            const emoName = data.emotion ?? "neutral";
            emoNameElement.textContent = `当前情绪：${EmoMap[emoName].emoName}`;
            emoDescElement.textContent = EmoMap[emoName].desc;
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            heartRateElement.textContent  = randHeartRate
            breathRateElement.textContent = randBreathRate;
        });
}


function updateMediaInfo() {
    const mediaCoverElement = document.querySelector('.media-cover');
    const titleElement = document.querySelector('.title');
    const artistGameElement = document.querySelector('.artist-game');
    const currentTimeElement = document.querySelector('.current-time');
    const totalTimeElement = document.querySelector('.total-time');

    // 模拟数据
    const mediaData = {
        type: 'music', // or 'game'
        title: '歌曲名/游戏名',
        artist: '艺术家/开发商',
        cover: 'https://via.placeholder.com/150', // 替换成真实的封面图片 URL
        duration: 245, // 总时长（秒）
        currentTime: 0 // 当前播放时间（秒）
    };

    mediaCoverElement.style.backgroundImage = `url(${mediaData.cover})`;
    titleElement.textContent = mediaData.title;
    artistGameElement.textContent = mediaData.type === 'music' ? mediaData.artist : `by ${mediaData.artist}`;

    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    totalTimeElement.textContent = formatTime(mediaData.duration);

    // 模拟播放进度
    setInterval(() => {
        if (mediaData.currentTime < mediaData.duration) {
            mediaData.currentTime++;
            currentTimeElement.textContent = formatTime(mediaData.currentTime);
        }
    }, 1000);
}

// 初始化
updateTime();
updateData();
// updateMediaInfo();

// 定时更新
setInterval(updateTime, 1000);
setInterval(updateData, 1000);