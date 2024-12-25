const baseURL = 'http://localhost:55555';

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

// 模拟数据更新 (实际应用中需要从真实数据源获取)
function updateHealthData() {
    const heartRateElement = document.querySelector('.heart-rate .value');
    const breathRateElement = document.querySelector('.breath-rate .value');

    const randHeartRate = Math.floor(Math.random() * 20) + 60; // 60-80 bpm
    const randBreathRate = Math.floor(Math.random() * 4) + 14;  // 14-18 breaths/min

    const url = `${baseURL}/get_emotion`;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            heartRateElement.textContent = data.heartRate ?? randHeartRate;
            breathRateElement.textContent = data.breathRate ?? randBreathRate;
        })
        .catch(error => {
            console.error('Error fetching weather data:', error);
            heartRateElement.textContent  = randHeartRate
            breathRateElement.textContent = randBreathRate;
        });
}

function updateEmotionData() {
    const weatherIconElement = document.querySelector('.weather-icon');
    const temperatureElement = document.querySelector('.temperature');
    const descriptionElement = document.querySelector('.description');
    const locationElement = document.querySelector('.location');

    const url = `${baseURL}/get_emotion`;

    fetch(url)
        .then(response => response.json())
        .then(data => {
            const iconCode = data.weather[0].icon;
            const iconUrl = `http://openweathermap.org/img/wn/${iconCode}@2x.png`;

            weatherIconElement.innerHTML = `<img src="${iconUrl}" alt="Weather Icon">`;
            temperatureElement.textContent = `${Math.round(data.main.temp)}°C`;
            descriptionElement.textContent = data.weather[0].description;
            locationElement.textContent = data.name;
        })
        .catch(error => {
            console.error('Error fetching weather data:', error);
            weatherIconElement.innerHTML = `<i class="fas fa-question-circle"></i>`;
            temperatureElement.textContent = `--°C`;
            descriptionElement.textContent = '获取失败';
            locationElement.textContent = '未知';
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
updateHealthData();
updateWeatherData();
updateMediaInfo();

// 定时更新
setInterval(updateTime, 1000);
setInterval(updateHealthData, 200); // 5 秒更新一次健康数据
// 天气数据更新频率可以根据需要调整，例如每 10 分钟更新一次
setInterval(updateWeatherData, 600000);