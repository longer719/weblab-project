/**
 * 植物病害检测可视化工具
 * 提供病害区域可视化和交互式结果分析功能
 */

// 可视化工具命名空间
const PlantVis = (function() {
    // 私有配置
    const config = {
        colors: {
            healthy: '#4CAF50',
            mild: '#FFC107', 
            moderate: '#FF9800',
            severe: '#F44336',
            highlight: 'rgba(255, 255, 0, 0.3)',
            boxDefault: '#2196F3',
            boxHighlight: '#FF4081'
        },
        opacity: 0.6,
        lineWidth: 2,
        fontSize: 14,
        fontFamily: 'Arial, sans-serif',
        animationDuration: 300,
        defaultScoreThreshold: 0.5
    };

    // 缓存DOM元素
    let elements = {};
    
    /**
     * 初始化可视化组件
     * @param {Object} options - 配置选项
     */
    function initialize(options = {}) {
        // 合并配置
        Object.assign(config, options);
        
        // 初始化DOM元素缓存
        cacheElements();
        
        // 设置事件监听器
        setupEventListeners();
        
        console.log('植物病害可视化组件初始化完成');
    }
    
    /**
     * 缓存常用DOM元素
     */
    function cacheElements() {
        elements.resultContainer = document.getElementById('results-container');
        elements.visualContainer = document.getElementById('visualization-container');
        elements.canvas = document.getElementById('vis-canvas');
        elements.controlPanel = document.getElementById('vis-controls');
    }
    
    /**
     * 设置事件监听
     */
    function setupEventListeners() {
        // 当可视化容器存在时
        if (elements.controlPanel) {
            // 监听阈值调整
            const thresholdSlider = elements.controlPanel.querySelector('.threshold-slider');
            if (thresholdSlider) {
                thresholdSlider.addEventListener('input', function(e) {
                    updateVisualizationThreshold(parseFloat(e.target.value));
                });
            }
            
            // 监听可视化模式切换
            const visModeSwitches = elements.controlPanel.querySelectorAll('.vis-mode-switch');
            visModeSwitches.forEach(switchEl => {
                switchEl.addEventListener('click', function(e) {
                    changeVisualizationMode(e.target.dataset.mode);
                });
            });
        }
    }
    
    /**
     * 渲染检测框
     * @param {HTMLImageElement|HTMLCanvasElement} image - 原始图像元素
     * @param {Array} detections - 检测结果数组
     * @param {Object} options - 渲染选项
     * @returns {HTMLCanvasElement} 渲染后的画布
     */
    function renderDetectionBoxes(image, detections, options = {}) {
        // 默认选项
        const opts = {
            scoreThreshold: options.scoreThreshold || config.defaultScoreThreshold,
            showLabels: options.showLabels !== undefined ? options.showLabels : true,
            showScores: options.showScores !== undefined ? options.showScores : true,
            highlightIndex: options.highlightIndex || -1,
            boxColors: options.boxColors || null
        };
        
        // 创建画布
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        
        const ctx = canvas.getContext('2d');
        
        // 绘制原始图像
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        // 过滤分数低于阈值的检测结果
        const filteredDetections = detections.filter(det => det.score >= opts.scoreThreshold);
        
        // 绘制每个检测框
        filteredDetections.forEach((det, index) => {
            const [x1, y1, x2, y2] = det.box;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // 确定框的颜色
            let boxColor;
            if (opts.boxColors && opts.boxColors[index]) {
                // 使用提供的颜色
                boxColor = opts.boxColors[index];
            } else if (index === opts.highlightIndex) {
                // 高亮选中的框
                boxColor = config.colors.boxHighlight;
            } else if (det.class_name && det.class_name.toLowerCase().includes('健康')) {
                // 健康样本用绿色
                boxColor = config.colors.healthy;
            } else {
                // 根据置信度确定颜色
                if (det.severity === 'severe') {
                    boxColor = config.colors.severe;
                } else if (det.severity === 'moderate') {
                    boxColor = config.colors.moderate;
                } else if (det.severity === 'mild') {
                    boxColor = config.colors.mild;
                } else {
                    // 默认蓝色
                    boxColor = config.colors.boxDefault;
                }
            }
            
            // 绘制边界框
            ctx.strokeStyle = boxColor;
            ctx.lineWidth = config.lineWidth;
            ctx.strokeRect(x1, y1, width, height);
            
            // 添加半透明填充
            ctx.fillStyle = boxColor + Math.floor(config.opacity * 255).toString(16).padStart(2, '0');
            ctx.fillRect(x1, y1, width, height);
            
            // 如果需要显示标签
            if (opts.showLabels || opts.showScores) {
                const label = det.class_name || '';
                const score = det.score ? Math.round(det.score * 100) + '%' : '';
                
                let displayText = '';
                if (opts.showLabels && opts.showScores) {
                    displayText = `${label}: ${score}`;
                } else if (opts.showLabels) {
                    displayText = label;
                } else if (opts.showScores) {
                    displayText = score;
                }
                
                if (displayText) {
                    // 绘制标签背景
                    ctx.font = `${config.fontSize}px ${config.fontFamily}`;
                    const textWidth = ctx.measureText(displayText).width;
                    const textHeight = config.fontSize;
                    const textX = x1;
                    const textY = y1 - textHeight - 5;
                    
                    ctx.fillStyle = boxColor;
                    ctx.fillRect(textX, textY, textWidth + 6, textHeight + 4);
                    
                    // 绘制标签文本
                    ctx.fillStyle = '#FFFFFF';
                    ctx.fillText(displayText, textX + 3, textY + textHeight);
                }
            }
        });
        
        return canvas;
    }

    /**
     * 渲染热图
     * @param {HTMLImageElement|HTMLCanvasElement} image - 原始图像
     * @param {Array} heatmapData - 热图数据
     * @param {Object} options - 渲染选项
     * @returns {HTMLCanvasElement} 渲染后的热图画布
     */
    function renderHeatmap(image, heatmapData, options = {}) {
        // 默认选项
        const opts = {
            opacity: options.opacity || 0.7,
            radius: options.radius || 20,
            blur: options.blur || 15,
            gradient: options.gradient || {
                0.4: 'blue',
                0.6: 'cyan',
                0.7: 'lime',
                0.8: 'yellow',
                1.0: 'red'
            }
        };
        
        // 创建画布
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        
        const ctx = canvas.getContext('2d');
        
        // 绘制原始图像
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        // 使用heatmap.js库生成热图
        if (window.h337) {
            // 创建临时容器
            const tempContainer = document.createElement('div');
            tempContainer.style.width = `${canvas.width}px`;
            tempContainer.style.height = `${canvas.height}px`;
            tempContainer.style.position = 'absolute';
            tempContainer.style.left = '-9999px';
            document.body.appendChild(tempContainer);
            
            // 创建热图实例
            const heatmapInstance = window.h337.create({
                container: tempContainer,
                radius: opts.radius,
                maxOpacity: opts.opacity,
                minOpacity: 0,
                blur: opts.blur,
                gradient: opts.gradient
            });
            
            // 设置数据
            heatmapInstance.setData({
                max: 1,
                data: heatmapData
            });
            
            // 获取热图canvas
            const heatCanvas = tempContainer.querySelector('canvas');
            ctx.drawImage(heatCanvas, 0, 0);
            
            // 清理临时DOM
            document.body.removeChild(tempContainer);
        } else {
            // 如果没有heatmap.js，使用简单的圆圈表示热点
            heatmapData.forEach(point => {
                const radius = Math.sqrt(point.value) * 50;
                ctx.beginPath();
                ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255, 0, 0, ${point.value * 0.7})`;
                ctx.fill();
            });
        }
        
        return canvas;
    }

    /**
     * 在指定容器中显示可视化内容
     * @param {HTMLElement} container - 容器元素
     * @param {HTMLCanvasElement} canvas - 绘制好的画布
     * @param {Object} options - 显示选项
     */
    function displayVisualization(container, canvas, options = {}) {
        if (!container || !canvas) return;
        
        // 清空容器
        container.innerHTML = '';
        
        // 添加画布
        container.appendChild(canvas);
        
        // 如果启用下载选项，添加下载按钮
        if (options.allowDownload) {
            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'download-btn';
            downloadBtn.innerHTML = '<i class="fas fa-download"></i> 下载分析结果';
            downloadBtn.addEventListener('click', () => {
                const link = document.createElement('a');
                link.download = '病害检测结果_' + new Date().toISOString().slice(0, 10) + '.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            });
            
            container.appendChild(downloadBtn);
        }
        
        // 如果需要添加控制面板
        if (options.showControls || options.enableThreshold || options.enableModeSwitch) {
            createControlPanel(container, {
                enableThreshold: options.enableThreshold || options.allowThresholdChange,
                threshold: options.initialThreshold || config.defaultScoreThreshold,
                enableModeSwitch: options.enableModeSwitch,
                currentMode: options.currentMode || 'boxes'
            });
        }
    }

    /**
     * 创建控制面板
     * @param {HTMLElement} container - 容器元素
     * @param {Object} options - 控制面板选项
     */
    function createControlPanel(container, options) {
        const controlPanel = document.createElement('div');
        controlPanel.className = 'vis-controls';
        controlPanel.id = 'vis-controls';
        
        // 添加阈值滑块
        if (options.enableThreshold) {
            const thresholdControl = document.createElement('div');
            thresholdControl.className = 'threshold-control';
            
            const label = document.createElement('label');
            label.textContent = '置信度阈值: ';
            
            const value = document.createElement('span');
            value.className = 'threshold-value';
            value.textContent = options.threshold || config.defaultScoreThreshold;
            
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.className = 'threshold-slider';
            slider.min = '0';
            slider.max = '1';
            slider.step = '0.01';
            slider.value = options.threshold || config.defaultScoreThreshold;
            
            slider.addEventListener('input', function() {
                value.textContent = this.value;
                // 触发自定义事件
                document.dispatchEvent(new CustomEvent('visualization-threshold-update', {
                    detail: { threshold: parseFloat(this.value) }
                }));
            });
            
            thresholdControl.appendChild(label);
            thresholdControl.appendChild(slider);
            thresholdControl.appendChild(value);
            controlPanel.appendChild(thresholdControl);
        }
        
        // 添加可视化模式选择
        if (options.enableModeSwitch) {
            const modeControl = document.createElement('div');
            modeControl.className = 'mode-control';
            
            const modeLabel = document.createElement('div');
            modeLabel.textContent = '可视化模式:';
            modeControl.appendChild(modeLabel);
            
            const modeOptions = [
                { id: 'boxes', text: '边界框', icon: 'fa-square-o' },
                { id: 'heatmap', text: '热图', icon: 'fa-fire' },
                { id: 'blend', text: '混合', icon: 'fa-object-group' }
            ];
            
            const modeButtons = document.createElement('div');
            modeButtons.className = 'mode-buttons';
            
            modeOptions.forEach(mode => {
                const btn = document.createElement('button');
                btn.className = `vis-mode-switch ${mode.id === (options.currentMode || 'boxes') ? 'active' : ''}`;
                btn.dataset.mode = mode.id;
                btn.innerHTML = `<i class="fas ${mode.icon}"></i> ${mode.text}`;
                
                btn.addEventListener('click', function() {
                    // 更新按钮状态
                    document.querySelectorAll('.vis-mode-switch').forEach(el => {
                        el.classList.remove('active');
                    });
                    this.classList.add('active');
                    
                    // 触发模式更改事件
                    document.dispatchEvent(new CustomEvent('mode-change', {
                        detail: { mode: mode.id }
                    }));
                });
                
                modeButtons.appendChild(btn);
            });
            
            modeControl.appendChild(modeButtons);
            controlPanel.appendChild(modeControl);
        }
        
        container.appendChild(controlPanel);
    }

    /**
     * 更新可视化阈值
     * @param {number} threshold - 新的阈值值
     */
    function updateVisualizationThreshold(threshold) {
        if (typeof threshold !== 'number' || threshold < 0 || threshold > 1) {
            console.error('无效的阈值:', threshold);
            return;
        }
        
        // 触发阈值更新事件
        document.dispatchEvent(new CustomEvent('visualization-threshold-update', {
            detail: { threshold }
        }));
    }

    /**
     * 更改可视化模式
     * @param {string} mode - 可视化模式 ('boxes', 'heatmap', 'blend')
     */
    function changeVisualizationMode(mode) {
        // 触发模式更改事件
        document.dispatchEvent(new CustomEvent('mode-change', {
            detail: { mode }
        }));
    }

    /**
     * 清除所有可视化内容
     */
    function clearVisualizations() {
        if (elements.visualContainer) {
            elements.visualContainer.innerHTML = '';
        }
    }

    /**
     * 将检测结果转换为热图数据
     * @param {Array} detections - 检测结果数组
     * @param {number} imageWidth - 图像宽度
     * @param {number} imageHeight - 图像高度
     * @returns {Array} 热图数据点数组
     */
    function detectionsToHeatmap(detections, imageWidth, imageHeight) {
        const heatmapData = [];
        
        detections.forEach(det => {
            if (det.score < 0.3) return; // 忽略低置信度检测
            
            const [x1, y1, x2, y2] = det.box;
            const centerX = (x1 + x2) / 2;
            const centerY = (y1 + y2) / 2;
            const width = x2 - x1;
            const height = y2 - y1;
            const area = width * height;
            
            // 主点 - 在中心位置添加强度最高的点
            heatmapData.push({
                x: centerX,
                y: centerY,
                value: det.score
            });
            
            // 边界点 - 在框的四个角和四条边的中点添加较弱的点
            const edgePoints = [
                [x1, y1], // 左上
                [x2, y1], // 右上
                [x1, y2], // 左下
                [x2, y2], // 右下
                [(x1 + x2) / 2, y1], // 上中
                [(x1 + x2) / 2, y2], // 下中
                [x1, (y1 + y2) / 2], // 左中
                [x2, (y1 + y2) / 2]  // 右中
            ];
            
            edgePoints.forEach(([x, y]) => {
                heatmapData.push({
                    x: x,
                    y: y,
                    value: det.score * 0.7 // 边界点强度为中心点的70%
                });
            });
            
            // 对于较大的区域，在内部添加更多点
            if (area > 10000) { // 根据实际情况调整阈值
                const stepX = width / 4;
                const stepY = height / 4;
                
                for (let i = 1; i < 4; i++) {
                    for (let j = 1; j < 4; j++) {
                        if (i === 2 && j === 2) continue; // 跳过中心点(已添加)
                        
                        const pointX = x1 + i * stepX;
                        const pointY = y1 + j * stepY;
                        
                        // 随机化位置和强度，使热图更自然
                        const jitter = 0.1; // 抖动范围
                        const jitteredX = pointX + (Math.random() * 2 - 1) * jitter * stepX;
                        const jitteredY = pointY + (Math.random() * 2 - 1) * jitter * stepY;
                        
                        heatmapData.push({
                            x: jitteredX,
                            y: jitteredY,
                            value: det.score * (0.5 + Math.random() * 0.3) // 随机强度
                        });
                    }
                }
            }
        });
        
        return heatmapData;
    }

    // 公开API
    return {
        initialize: initialize,
        renderDetectionBoxes: renderDetectionBoxes,
        renderHeatmap: renderHeatmap,
        displayVisualization: displayVisualization,
        updateVisualizationThreshold: updateVisualizationThreshold,
        changeVisualizationMode: changeVisualizationMode,
        clearVisualizations: clearVisualizations,
        detectionsToHeatmap: detectionsToHeatmap
    };
})();

// 页面加载后初始化可视化组件
document.addEventListener('DOMContentLoaded', function() {
    // 当页面加载完成时，只初始化可视化工具，但不立即显示
    // 等待main.js中的应用逻辑调用相关方法显示可视化结果
    PlantVis.initialize();
    
    // 监听自定义事件，这些事件由main.js中的代码触发
    document.addEventListener('plant-detection-result', function(e) {
        // 如果有检测结果，显示可视化
        if (e.detail && e.detail.image && e.detail.detections) {
            const visContainer = document.getElementById('visualization-container');
            if (visContainer) {
                const canvas = PlantVis.renderDetectionBoxes(
                    e.detail.image, 
                    e.detail.detections,
                    { showLabels: true, showScores: true }
                );
                
                PlantVis.displayVisualization(visContainer, canvas, {
                    allowThresholdChange: true,
                    initialThreshold: 0.5,
                    visModes: [
                        { label: '边界框', value: 'boxes' },
                        { label: '热图', value: 'heatmap' }
                    ],
                    currentMode: 'boxes'
                });
            }
        }
    });
});