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
            showLabels: options.showLabels !== false,
            showScores: options.showScores !== false,
            highlightIndex: options.highlightIndex
        };
        
        // 创建画布
        const canvas = document.createElement('canvas');
        canvas.width = image.naturalWidth || image.width;
        canvas.height = image.naturalHeight || image.height;
        const ctx = canvas.getContext('2d');
        
        // 绘制原图
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        // 过滤检测结果，仅显示高于阈值的
        const validDetections = detections.filter(det => det.score >= opts.scoreThreshold);
        
        // 绘制每个检测框
        validDetections.forEach((detection, index) => {
            const { box, score, class_name } = detection;
            
            // 设置样式
            const isHighlighted = index === opts.highlightIndex;
            ctx.strokeStyle = isHighlighted ? config.colors.boxHighlight : config.colors.boxDefault;
            ctx.lineWidth = isHighlighted ? config.lineWidth + 1 : config.lineWidth;
            
            // 根据严重程度调整颜色
            if (detection.severity) {
                switch(detection.severity.toLowerCase()) {
                    case 'severe': ctx.strokeStyle = config.colors.severe; break;
                    case 'moderate': ctx.strokeStyle = config.colors.moderate; break;
                    case 'mild': ctx.strokeStyle = config.colors.mild; break;
                    case 'healthy': ctx.strokeStyle = config.colors.healthy; break;
                }
            }
            
            // 绘制矩形
            ctx.beginPath();
            ctx.rect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
            ctx.stroke();
            
            // 标签背景
            if (opts.showLabels || opts.showScores) {
                const label = opts.showLabels ? class_name : '';
                const scoreText = opts.showScores ? `${(score * 100).toFixed(0)}%` : '';
                const displayText = [label, scoreText].filter(Boolean).join(': ');
                
                if (displayText) {
                    const textWidth = ctx.measureText(displayText).width + 10;
                    const textHeight = config.fontSize + 10;
                    
                    ctx.fillStyle = isHighlighted ? 
                        config.colors.boxHighlight : 
                        config.colors.boxDefault;
                    ctx.globalAlpha = 0.7;
                    ctx.fillRect(box[0], box[1] - textHeight, textWidth, textHeight);
                    ctx.globalAlpha = 1.0;
                    
                    // 标签文本
                    ctx.fillStyle = 'white';
                    ctx.font = `${config.fontSize}px ${config.fontFamily}`;
                    ctx.fillText(displayText, box[0] + 5, box[1] - 5);
                }
            }
            
            // 添加检测框点击事件数据
            if (detection.treatment_info) {
                // 存储治疗信息到canvas元素的自定义数据中
                if (!canvas.detectionData) {
                    canvas.detectionData = [];
                }
                canvas.detectionData.push({
                    box: [box[0], box[1], box[2], box[3]],
                    treatment: detection.treatment_info
                });
            }
        });
        
        // 添加点击事件监听
        if (!canvas.hasClickListener && canvas.detectionData && canvas.detectionData.length > 0) {
            canvas.addEventListener('click', function(e) {
                // 获取点击位置相对于canvas的坐标
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                // 缩放比例
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                
                // 转换为canvas坐标
                const canvasX = x * scaleX;
                const canvasY = y * scaleY;
                
                // 检查是否点击了某个检测框
                for (const data of canvas.detectionData) {
                    const [x1, y1, x2, y2] = data.box;
                    if (canvasX >= x1 && canvasX <= x2 && canvasY >= y1 && canvasY <= y2) {
                        // 显示该框对应的治疗信息
                        if (typeof showDetailedTreatment === 'function' && data.treatment) {
                            showDetailedTreatment(data.treatment);
                        }
                        break;
                    }
                }
            });
            canvas.hasClickListener = true;
        }
        
        return canvas;
    }
    
    /**
     * 生成热图并叠加在原图上
     * @param {HTMLImageElement|HTMLCanvasElement} image - 原始图像元素
     * @param {Array} heatmapData - 热图数据
     * @param {Object} options - 渲染选项
     * @returns {HTMLCanvasElement} 渲染后的画布
     */
    function renderHeatmap(image, heatmapData, options = {}) {
        // 默认选项
        const opts = {
            opacity: options.opacity || config.opacity,
            colorScale: options.colorScale || ['blue', 'lime', 'yellow', 'red'],
            blur: options.blur || 15
        };
        
        // 创建画布
        const canvas = document.createElement('canvas');
        canvas.width = image.naturalWidth || image.width;
        canvas.height = image.naturalHeight || image.height;
        const ctx = canvas.getContext('2d');
        
        // 绘制原图
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        // 检查我们是否有热图数据
        if (!heatmapData || !heatmapData.length) return canvas;
        
        // 创建热图数据
        // 注意：在实际应用中，这里可能会使用类似heatmap.js的库
        // 由于我们只提供架构，这里用简化的方法
        const heatmapCanvas = document.createElement('canvas');
        heatmapCanvas.width = canvas.width;
        heatmapCanvas.height = canvas.height;
        const heatCtx = heatmapCanvas.getContext('2d');
        
        // 绘制热点
        heatmapData.forEach(point => {
            const gradient = heatCtx.createRadialGradient(
                point.x, point.y, 0, 
                point.x, point.y, point.radius || 30
            );
            
            gradient.addColorStop(0, `rgba(255, 0, 0, ${point.value})`);
            gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
            
            heatCtx.fillStyle = gradient;
            heatCtx.beginPath();
            heatCtx.arc(point.x, point.y, point.radius || 30, 0, Math.PI * 2);
            heatCtx.fill();
        });
        
        // 应用模糊效果增强热图外观
        if (heatCtx.filter) {
            heatCtx.filter = `blur(${opts.blur}px)`;
            heatCtx.drawImage(heatmapCanvas, 0, 0);
            heatCtx.filter = 'none';
        }
        
        // 叠加热图到原图
        ctx.globalAlpha = opts.opacity;
        ctx.drawImage(heatmapCanvas, 0, 0);
        ctx.globalAlpha = 1.0;
        
        return canvas;
    }
    
    /**
     * 生成并显示GradCAM可视化结果
     * @param {HTMLImageElement} image - 原始图像元素
     * @param {Array} gradcamData - GradCAM数据
     * @param {Object} options - 渲染选项
     */
    function visualizeGradCAM(image, gradcamData, options = {}) {
        // 转换GradCAM数据为热图数据格式
        const heatmapData = convertGradCAMToHeatmap(gradcamData);
        
        // 渲染热图
        return renderHeatmap(image, heatmapData, {
            opacity: 0.7,
            ...options
        });
    }
    
    /**
     * 转换GradCAM数据为热图数据格式
     * @param {Array|Object} gradcamData - GradCAM数据
     * @returns {Array} 热图数据
     */
    function convertGradCAMToHeatmap(gradcamData) {
        // 实际应用中，这里会有具体的转换逻辑
        // 这里只是演示用的简化实现
        if (Array.isArray(gradcamData)) {
            return gradcamData.map(point => ({
                x: point.x,
                y: point.y,
                value: point.activation,
                radius: 20
            }));
        } else if (gradcamData && gradcamData.heatmap) {
            // 如果直接提供了热图数据
            return gradcamData.heatmap;
        }
        
        return [];
    }
    
    /**
     * 在给定容器中显示可视化结果
     * @param {HTMLElement} container - 目标容器元素
     * @param {HTMLCanvasElement} canvas - 可视化画布
     * @param {Object} options - 显示选项
     */
    function displayVisualization(container, canvas, options = {}) {
        if (!container || !canvas) return;
        
        // 清除现有内容
        container.innerHTML = '';
        
        // 设置画布样式
        canvas.style.maxWidth = '100%';
        canvas.style.height = 'auto';
        canvas.style.borderRadius = '4px';
        canvas.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
        
        // 添加到容器
        container.appendChild(canvas);
        
        // 如果需要添加交互控件
        if (options.interactive !== false) {
            addInteractiveControls(container, options);
        }
    }
    
    /**
     * 添加交互控件
     * @param {HTMLElement} container - 目标容器
     * @param {Object} options - 控件选项
     */
    function addInteractiveControls(container, options = {}) {
        const controlPanel = document.createElement('div');
        controlPanel.className = 'vis-control-panel';
        controlPanel.style.marginTop = '10px';
        controlPanel.style.padding = '10px';
        controlPanel.style.backgroundColor = '#f5f5f5';
        controlPanel.style.borderRadius = '4px';
        
        // 添加阈值滑块
        if (options.allowThresholdChange !== false) {
            const thresholdControl = createThresholdControl(options.initialThreshold || 0.5);
            controlPanel.appendChild(thresholdControl);
        }
        
        // 添加可视化模式切换
        if (options.visModes && options.visModes.length) {
            const modeControl = createModeControl(options.visModes, options.currentMode);
            controlPanel.appendChild(modeControl);
        }
        
        // 添加下载按钮
        if (options.allowDownload !== false) {
            const downloadBtn = createDownloadButton(container.querySelector('canvas'));
            controlPanel.appendChild(downloadBtn);
        }
        
        // 添加到容器
        container.appendChild(controlPanel);
    }
    
    /**
     * 创建阈值控制滑块
     * @param {number} initialValue - 初始阈值
     * @returns {HTMLElement} 滑块元素
     */
    function createThresholdControl(initialValue = 0.5) {
        const container = document.createElement('div');
        container.className = 'threshold-control';
        container.style.marginBottom = '10px';
        
        const label = document.createElement('label');
        label.textContent = '检测阈值: ';
        label.style.display = 'block';
        label.style.marginBottom = '5px';
        label.style.fontWeight = 'bold';
        
        const valueDisplay = document.createElement('span');
        valueDisplay.textContent = `${(initialValue * 100).toFixed(0)}%`;
        valueDisplay.style.marginLeft = '5px';
        valueDisplay.className = 'threshold-value';
        label.appendChild(valueDisplay);
        
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = '1';
        slider.step = '0.01';
        slider.value = initialValue;
        slider.className = 'threshold-slider';
        slider.style.width = '100%';
        
        // 更新显示值
        slider.addEventListener('input', function() {
            valueDisplay.textContent = `${(this.value * 100).toFixed(0)}%`;
            // 触发自定义事件，让外部知道阈值变化
            const event = new CustomEvent('threshold-change', { detail: { value: parseFloat(this.value) } });
            document.dispatchEvent(event);
        });
        
        container.appendChild(label);
        container.appendChild(slider);
        
        return container;
    }
    
    /**
     * 创建模式切换控制
     * @param {Array} modes - 可用模式
     * @param {string} currentMode - 当前模式
     * @returns {HTMLElement} 模式切换元素
     */
    function createModeControl(modes, currentMode) {
        const container = document.createElement('div');
        container.className = 'mode-control';
        container.style.marginBottom = '10px';
        
        const label = document.createElement('div');
        label.textContent = '可视化模式:';
        label.style.marginBottom = '5px';
        label.style.fontWeight = 'bold';
        
        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'mode-buttons';
        buttonsContainer.style.display = 'flex';
        buttonsContainer.style.gap = '5px';
        
        modes.forEach(mode => {
            const button = document.createElement('button');
            button.textContent = mode.label || mode;
            button.dataset.mode = mode.value || mode;
            button.className = 'vis-mode-switch';
            button.style.padding = '5px 10px';
            button.style.border = '1px solid #ccc';
            button.style.borderRadius = '4px';
            button.style.backgroundColor = (mode.value || mode) === currentMode ? '#2196F3' : '#ffffff';
            button.style.color = (mode.value || mode) === currentMode ? '#ffffff' : '#000000';
            button.style.cursor = 'pointer';
            
            button.addEventListener('click', function() {
                // 更新按钮状态
                buttonsContainer.querySelectorAll('button').forEach(btn => {
                    btn.style.backgroundColor = '#ffffff';
                    btn.style.color = '#000000';
                });
                this.style.backgroundColor = '#2196F3';
                this.style.color = '#ffffff';
                
                // 触发自定义事件
                const event = new CustomEvent('mode-change', { 
                    detail: { mode: this.dataset.mode } 
                });
                document.dispatchEvent(event);
            });
            
            buttonsContainer.appendChild(button);
        });
        
        container.appendChild(label);
        container.appendChild(buttonsContainer);
        
        return container;
    }
    
    /**
     * 创建下载按钮
     * @param {HTMLCanvasElement} canvas - 要下载的画布
     * @returns {HTMLElement} 下载按钮
     */
    function createDownloadButton(canvas) {
        const button = document.createElement('button');
        button.textContent = '下载可视化结果';
        button.className = 'download-vis-btn';
        button.style.marginTop = '10px';
        button.style.padding = '8px 16px';
        button.style.backgroundColor = '#4CAF50';
        button.style.color = 'white';
        button.style.border = 'none';
        button.style.borderRadius = '4px';
        button.style.cursor = 'pointer';
        button.style.fontWeight = 'bold';
        
        button.addEventListener('click', function() {
            if (!canvas) return;
            
            // 创建下载链接
            const link = document.createElement('a');
            link.download = `plant-disease-visualization-${new Date().getTime()}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
        });
        
        return button;
    }
    
    /**
     * 创建和显示分析图表
     * @param {HTMLElement} container - 目标容器
     * @param {Object} data - 图表数据
     * @param {string} chartType - 图表类型
     * @returns {Object} 图表实例
     */
    function createChart(container, data, chartType = 'bar') {
        // 在实际应用中，这里会使用Chart.js或其他图表库
        // 这里我们只提供基本结构作为示例
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.style.width = '100%';
        chartContainer.style.height = '300px';
        chartContainer.style.marginTop = '20px';
        chartContainer.style.position = 'relative';
        
        // 添加一个图表说明
        const chartInfo = document.createElement('div');
        chartInfo.className = 'chart-info';
        chartInfo.textContent = `这里将显示${chartType}图表。在实际应用中会集成图表库。`;
        chartInfo.style.textAlign = 'center';
        chartInfo.style.padding = '20px';
        chartInfo.style.backgroundColor = '#f9f9f9';
        chartInfo.style.border = '1px dashed #ccc';
        chartInfo.style.borderRadius = '4px';
        
        chartContainer.appendChild(chartInfo);
        container.appendChild(chartContainer);
        
        return {
            update: function(newData) {
                console.log('更新图表数据:', newData);
                // 实际应用中这里会更新图表
            },
            destroy: function() {
                if (chartContainer.parentNode) {
                    chartContainer.parentNode.removeChild(chartContainer);
                }
            }
        };
    }
    
    /**
     * 更新可视化阈值
     * @param {number} threshold - 新阈值
     */
    function updateVisualizationThreshold(threshold) {
        // 触发自定义事件
        const event = new CustomEvent('visualization-threshold-update', {
            detail: { threshold }
        });
        document.dispatchEvent(event);
    }
    
    /**
     * 切换可视化模式
     * @param {string} mode - 可视化模式
     */
    function changeVisualizationMode(mode) {
        // 触发自定义事件
        const event = new CustomEvent('visualization-mode-change', {
            detail: { mode }
        });
        document.dispatchEvent(event);
    }
    
    // 公开API
    return {
        initialize,
        renderDetectionBoxes,
        renderHeatmap,
        visualizeGradCAM,
        displayVisualization,
        createChart,
        updateVisualizationThreshold,
        changeVisualizationMode
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