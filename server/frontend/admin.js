// 获取按钮和输入框元素
const heart_mean = document.getElementById('heart_mean');
const heart_freq = document.getElementById('heart_freq');
const heart_vary = document.getElementById('heart_vary');
const breath_mean = document.getElementById('breath_mean');
const breath_freq = document.getElementById('breath_freq');
const breath_vary = document.getElementById('breath_vary');

const heart_apply = document.getElementById('heart_apply');
const emo_buttons = document.querySelectorAll('.emo-btn');

// overwrite = (event_name, is_enable) {

// }

emo_buttons.forEach(button => {
    button.addEventListener('click', () => {
        console.log(button.id);
    });
});

// 应用按钮的点击事件
applyButton.addEventListener('click', () => {
    const value1 = input1.value;
    const value2 = input2.value;
    const value3 = input3.value;

    // 在这里处理应用逻辑，例如发送数据到服务器或更新配置
    console.log('应用参数:', value1, value2, value3);
    alert(`参数已应用: ${value1}, ${value2}, ${value3}`);
});

// 重置按钮的点击事件
resetButton.addEventListener('click', () => {
    input1.value = 25; // 默认值
    input2.value = 50;
    input3.value = 75;

    console.log('参数已重置');
    alert('参数已重置');
});

// 特殊操作按钮的点击事件
specialButton.addEventListener('click', () => {
    // 在这里添加特殊操作的逻辑
    console.log('执行特殊操作');
    alert('执行特殊操作');
});