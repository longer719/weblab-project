/**
 * 植物病害检测可视化工具
 * 提供病害区域可视化和交互式结果分析功能
 * 优化版：增强模块化设计，减少事件监听器数量
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
    
    // 可视化状态
    let state = {
        currentMode: 'boxes',
        threshold: config.defaultScoreThreshold,
        currentDetections: null,
        originalImage: null,
        lastRenderedCanvas: null
    };
    
    /**
     * 初始化可视化组件
     * @param {Object} options - 配置选项
     */
    function initialize(options = {}) {
        // 合并配置
        Object.assign(config, options);
        
        // 初始化DOM元素缓存
        cacheElements();
        
        // 使用事件委托设置事件监听器
        setupEventDelegation();
        
        console.log('植物病害可视化组件初始化完成');
    }
    
    /**
     * 缓存常用DOM元素
     */
    function cacheElements() {
        elements.resultContainer = document.getElementById('results-container');
        elements.visualContainer = document.getElementById('visualization-container');
    }
    
    /**
     * 设置事件委托
     * 使用委托模式减少事件监听器数量
     */
    function setupEventDelegation() {
        // 使用事件委托为可视化容器添加事件监听
        document.body.addEventListener('click', function(e) {
            // 处理阈值滑块点击
            if (e.target && e.target.classList.contains('threshold-slider')) {
                return; // 滑块有自己的input事件处理
            }
            
            // 处理模式切换按钮点击
            if (e.target && (e.target.classList.contains('vis-mode-switch') || 
                (e.target.parentElement && e.target.parentElement.classList.contains('vis-mode-switch')))) {
                
                const button = e.target.classList.contains('vis-mode-switch') ? 
                    e.target : e.target.parentElement;
                const mode = button.dataset.mode;
                
                if (mode && mode !== state.currentMode) {
                    console.log('可视化模式切换:', mode);
                    
                    // 更新UI
                    document.querySelectorAll('.vis-mode-switch').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    button.classList.add('active');
                    
                    // 更新状态并触发事件
                    state.currentMode = mode;
                    changeVisualizationMode(mode);
                }
            }
            
            // 处理下载按钮点击
            if (e.target && (e.target.classList.contains('download-btn') || 
                (e.target.parentElement && e.target.parentElement.classList.contains('download-btn')))) {
                if (state.lastRenderedCanvas) {
                    console.log('下载可视化结果');
                    const link = document.createElement('a');
                    link.download = '病害检测结果_' + new Date().toISOString().slice(0, 10) + '.png';
                    link.href = state.lastRenderedCanvas.toDataURL('image/png');
                    link.click();
                }
            }
        });
        
        // 为阈值滑块添加委托事件（需要input事件）
        document.body.addEventListener('input', function(e) {
            if (e.target && e.target.classList.contains('threshold-slider')) {
                const threshold = parseFloat(e.target.value);
                console.log('阈值调整:', threshold);
                
                // 更新显示值
                const valueDisplay = e.target.parentElement.querySelector('.threshold-value');
                if (valueDisplay) {
                    valueDisplay.textContent = threshold.toFixed(2);
                }
                
                // 更新状态并触发事件
                state.threshold = threshold;
                updateVisualizationThreshold(threshold);
            }
        });
    }
    
    /**
     * 渲染检测框
     * @param {HTMLImageElement|HTMLCanvasElement} image - 原始图像元素
     * @param {Array} detections - 检测结果数组
     * @param {Object} options - 渲染选项
     * @returns {HTMLCanvasElement} 渲染后的画布
     */
    function renderDetectionBoxes(image, detections, options = {}) {
        // 保存状态供后续使用
        state.originalImage = image;
        state.currentDetections = detections;
        
        // 默认选项
        const opts = {
            scoreThreshold: options.scoreThreshold !== undefined ? options.scoreThreshold : state.threshold,
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
        
        // 调试输出
        console.log('绘制边界框，检测结果:', filteredDetections);
        
        // 绘制每个检测框
        filteredDetections.forEach((det, index) => {
            // 检查box格式，确保使用正确的坐标
            let x1, y1, x2, y2;
            
            if (Array.isArray(det.box)) {
                // 如果box是数组格式[x1,y1,x2,y2]
                [x1, y1, x2, y2] = det.box;
            } else if (det.box && typeof det.box === 'object') {
                // 如果box是对象格式{x,y,width,height}
                x1 = det.box.x || 0;
                y1 = det.box.y || 0;
                x2 = x1 + (det.box.width || 0);
                y2 = y1 + (det.box.height || 0);
            } else if (det.bbox) {
                // 尝试使用bbox属性
                x1 = det.bbox.x || 0;
                y1 = det.bbox.y || 0;
                x2 = x1 + (det.bbox.width || 0);
                y2 = y1 + (det.bbox.height || 0);
            } else {
                // 如果无法获取坐标，使用图像尺寸的20%作为默认检测框
                const defaultSize = Math.min(image.width, image.height) * 0.2;
                x1 = image.width/2 - defaultSize/2;
                y1 = image.height/2 - defaultSize/2;
                x2 = x1 + defaultSize;
                y2 = y1 + defaultSize;
                console.warn('未找到有效的边界框坐标，使用默认值');
            }
            
            // 检查坐标有效性
            x1 = Math.max(0, Math.min(x1, image.width));
            y1 = Math.max(0, Math.min(y1, image.height));
            x2 = Math.max(0, Math.min(x2, image.width));
            y2 = Math.max(0, Math.min(y2, image.height));
            
            const width = x2 - x1;
            const height = y2 - y1;
            
            // 确定框的颜色
            let boxColor;
            if (opts.boxColors && opts.boxColors[index]) {
                boxColor = opts.boxColors[index];
            } else if (index === opts.highlightIndex) {
                boxColor = config.colors.boxHighlight;
            } else if (det.class_name && det.class_name.toLowerCase().includes('健康')) {
                boxColor = config.colors.healthy;
            } else {
                // 根据严重程度确定颜色
                if (det.severity === 'severe') {
                    boxColor = config.colors.severe;
                } else if (det.severity === 'moderate') {
                    boxColor = config.colors.moderate;
                } else if (det.severity === 'mild') {
                    boxColor = config.colors.mild;
                } else {
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
                    
                    // 确保标签在图像内
                    let textX = x1;
                    let textY = y1 - textHeight - 5;
                    
                    // 如果标签会超出图像顶部，则放在框的内部顶部
                    if (textY < textHeight) {
                        textY = y1 + textHeight;
                    }
                    
                    ctx.fillStyle = boxColor;
                    ctx.fillRect(textX, textY - textHeight, textWidth + 6, textHeight + 4);
                    
                    // 绘制标签文本
                    ctx.fillStyle = '#FFFFFF';
                    ctx.fillText(displayText, textX + 3, textY);
                }
            }
        });
        
        // 保存渲染结果
        state.lastRenderedCanvas = canvas;
        
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
        // 保存状态
        state.originalImage = image;
        
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
        
        console.log('开始渲染热图, heatmap.js可用:', !!window.h337);
        
        // 使用heatmap.js库生成热图
        if (window.h337) {
            try {
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
                if (heatCanvas) {
                    ctx.drawImage(heatCanvas, 0, 0);
                    console.log('热图渲染成功');
                } else {
                    console.error('热图画布不存在');
                    fallbackHeatmapRender(ctx, heatmapData, canvas.width, canvas.height);
                }
                
                // 清理临时DOM
                document.body.removeChild(tempContainer);
            } catch (error) {
                console.error('热图渲染出错:', error);
                fallbackHeatmapRender(ctx, heatmapData, canvas.width, canvas.height);
            }
        } else {
            // 如果没有heatmap.js，使用简单的圆圈表示热点
            console.log('使用备用热图渲染方法');
            fallbackHeatmapRender(ctx, heatmapData, canvas.width, canvas.height);
        }
        
        // 保存渲染结果
        state.lastRenderedCanvas = canvas;
        
        return canvas;
    }

    /**
     * 备用热图渲染方法
     * 当heatmap.js不可用时使用简单圆圈表示热点
     */
    function fallbackHeatmapRender(ctx, heatmapData, width, height) {
        heatmapData.forEach(point => {
            const radius = point.radius || 20;
            const alpha = point.value || 0.5;
            
            ctx.beginPath();
            ctx.arc(point.x, point.y, radius, 0, Math.PI * 2, true);
            ctx.fillStyle = `rgba(255, 0, 0, ${alpha})`;
            ctx.fill();
        });
    }
    
    /**
     * 渲染混合模式（边界框+半透明热图）
     * 新增功能：将边界框和热图结合
     * @param {HTMLImageElement|HTMLCanvasElement} image - 原始图像
     * @param {Array} detections - 检测结果数组
     * @param {Object} options - 渲染选项
     * @returns {HTMLCanvasElement} 渲染后的画布
     */
    function renderBlendMode(image, detections, options = {}) {
        console.log("渲染混合可视化模式");
        
        // 保存状态
        state.originalImage = image;
        state.currentDetections = detections;
        
        // 首先绘制边界框
        const boxCanvas = renderDetectionBoxes(image, detections, options);
        
        // 过滤检测结果
        const filteredDetections = detections.filter(det => det.score >= (options.scoreThreshold || state.threshold));
        
        // 生成热图数据
        const heatmapData = detectionsToHeatmap(filteredDetections, image.width, image.height);
        
        // 创建最终画布
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext('2d');
        
        // 绘制边界框结果
        ctx.drawImage(boxCanvas, 0, 0);
        
        // 使用heatmap.js库生成热图
        if (window.h337) {
            // 创建临时容器
            const tempContainer = document.createElement('div');
            tempContainer.style.width = `${canvas.width}px`;
            tempContainer.style.height = `${canvas.height}px`;
            tempContainer.style.position = 'absolute';
            tempContainer.style.left = '-9999px';
            document.body.appendChild(tempContainer);
            
            // 配置热图
            const heatmapInstance = h337.create({
                container: tempContainer,
                radius: options.radius || 30,
                maxOpacity: 0.6,
                minOpacity: 0,
                blur: 0.75,
                gradient: options.gradient || {
                    0.4: 'blue',
                    0.6: 'cyan',
                    0.7: 'lime',
                    0.8: 'yellow',
                    1.0: 'red'
                }
            });
            
            // 设置热图数据
            heatmapInstance.setData({
                max: 1,
                data: heatmapData
            });
            
            // 获取热图画布
            const heatCanvas = tempContainer.querySelector('canvas');
            
            // 将热图叠加到原始画布上
            ctx.globalAlpha = 0.5;  // 半透明叠加
            ctx.drawImage(heatCanvas, 0, 0);
            ctx.globalAlpha = 1.0;
            
            // 移除临时容器
            document.body.removeChild(tempContainer);
        } else {
            console.warn('heatmap.js不可用，使用备选方法渲染');
            ctx.drawImage(boxCanvas, 0, 0);
        }
        
        // 保存渲染结果
        state.lastRenderedCanvas = canvas;
        
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
        
        // 添加画布和设置样式
        canvas.style.maxWidth = '100%';
        canvas.style.height = 'auto';
        container.appendChild(canvas);
        
        // 保存当前可视化状态
        state.lastRenderedCanvas = canvas;
        state.currentMode = options.currentMode || state.currentMode;
        state.threshold = options.initialThreshold || state.threshold;
        
        // 显示容器
        container.style.display = 'block';
        
        // 如果启用下载选项，添加下载按钮
        if (options.allowDownload) {
            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'download-btn';
            downloadBtn.innerHTML = '<i class="fas fa-download"></i> 下载分析结果';
            container.appendChild(downloadBtn);
        }
        
        // 如果需要添加控制面板
        if (options.showControls || options.enableThreshold || options.enableModeSwitch) {
            createControlPanel(container, {
                enableThreshold: options.enableThreshold || options.allowThresholdChange,
                threshold: options.initialThreshold || state.threshold,
                enableModeSwitch: options.enableModeSwitch,
                currentMode: options.currentMode || state.currentMode,
                visModes: options.visModes || [
                    { label: '边界框', value: 'boxes' },
                    { label: '热图', value: 'heatmap' },
                    { label: '混合', value: 'blend' }
                ]
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
            
            // 从选项中获取可用的模式或使用默认
            const modeOptions = options.visModes || [
                { label: '边界框', value: 'boxes', icon: 'fa-square-o' },
                { label: '热图', value: 'heatmap', icon: 'fa-fire' },
                { label: '混合', value: 'blend', icon: 'fa-object-group' }
            ];
            
            const modeButtons = document.createElement('div');
            modeButtons.className = 'mode-buttons';
            
            modeOptions.forEach(mode => {
                const btn = document.createElement('button');
                btn.className = `vis-mode-switch ${mode.value === (options.currentMode || 'boxes') ? 'active' : ''}`;
                btn.dataset.mode = mode.value;
                // 使用提供的图标或默认图标
                const icon = mode.icon || 'fa-image';
                btn.innerHTML = `<i class="fas ${icon}"></i> ${mode.label}`;
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
        
        // 更新状态
        state.threshold = threshold;
        
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
        if (!['boxes', 'heatmap', 'blend'].includes(mode)) {
            console.error('无效的可视化模式:', mode);
            return;
        }
        
        // 更新状态
        state.currentMode = mode;
        
        // 触发模式更改事件
        document.dispatchEvent(new CustomEvent('mode-change', {
            detail: { mode }
        }));
    }

    /**
     * 重新渲染当前可视化
     * 根据当前状态和模式刷新可视化
     * @param {HTMLElement} container - 要渲染到的容器
     */
    function refreshVisualization(container) {
        if (!container || !state.originalImage || !state.currentDetections) {
            return;
        }
        
        let canvas;
        
        // 根据当前模式选择渲染方法
        switch (state.currentMode) {
            case 'heatmap':
                const heatmapData = detectionsToHeatmap(
                    state.currentDetections.filter(d => d.score >= state.threshold),
                    state.originalImage.width,
                    state.originalImage.height
                );
                canvas = renderHeatmap(state.originalImage, heatmapData);
                break;
                
            case 'blend':
                canvas = renderBlendMode(state.originalImage, state.currentDetections, {
                    scoreThreshold: state.threshold
                });
                break;
                
            case 'boxes':
            default:
                canvas = renderDetectionBoxes(state.originalImage, state.currentDetections, {
                    scoreThreshold: state.threshold
                });
                break;
        }
        
        // 显示结果
        if (canvas) {
            // 保留控制面板的配置
            const controlPanelSettings = {
                enableThreshold: true,
                threshold: state.threshold,
                enableModeSwitch: true,
                currentMode: state.currentMode,
                showControls: true
            };
            
            displayVisualization(container, canvas, controlPanelSettings);
        }
    }
    
    /**
     * 清除所有可视化内容
     */
    function clearVisualizations() {
        // 清除容器内容
        if (elements.visualContainer) {
            elements.visualContainer.innerHTML = '';
        }
        
        // 重置状态
        state.lastRenderedCanvas = null;
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
        console.log(detections);
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
                value: det.score,
                radius: Math.max(width, height) * 0.25 // 动态调整半径
            });
            
            // 为较大区域添加更多点
            if (area > 5000) { // 阈值可以根据实际情况调整
                // 在框内添加随机分布的点
                const pointCount = Math.min(Math.floor(area / 2000), 15); // 最多15个额外点
                
                for (let i = 0; i < pointCount; i++) {
                    // 添加随机位置的点
                    const rx = x1 + Math.random() * width;
                    const ry = y1 + Math.random() * height;
                    
                    // 距离中心越远，值越小
                    const distanceToCenter = Math.sqrt(
                        Math.pow(rx - centerX, 2) + 
                        Math.pow(ry - centerY, 2)
                    );
                    const maxDistance = Math.sqrt(Math.pow(width/2, 2) + Math.pow(height/2, 2));
                    const distanceRatio = 1 - Math.min(distanceToCenter / maxDistance, 1);
                    
                    heatmapData.push({
                        x: rx,
                        y: ry,
                        value: det.score * distanceRatio * 0.8, // 根据距离衰减
                        radius: Math.max(width, height) * 0.15 // 较小的半径
                    });
                }
            }
        });
        
        return heatmapData;
    }

    // 公开API
    return {
        // 基础方法
        initialize,
        
        // 渲染方法
        renderDetectionBoxes,
        renderHeatmap,
        renderBlendMode,
        displayVisualization,
        
        // 控制方法
        updateVisualizationThreshold,
        changeVisualizationMode,
        refreshVisualization,
        clearVisualizations,
        
        // 工具方法
        detectionsToHeatmap,
        
        // 用于调试的状态访问 (生产环境通常不需要)
        getState: () => ({...state})
    };
})();

// 页面加载后初始化可视化组件
document.addEventListener('DOMContentLoaded', function() {
    // 初始化可视化工具
    PlantVis.initialize();
    
    // 监听自定义事件
    document.addEventListener('plant-detection-result', function(e) {
        console.log('接收到plant-detection-result事件:', e.detail);
        if (e.detail && e.detail.image && e.detail.detections) {
            const visContainer = document.getElementById('visualization-container');
            if (visContainer) {
                // 渲染检测框
                const canvas = PlantVis.renderDetectionBoxes(
                    e.detail.image, 
                    e.detail.detections,
                    { showLabels: true, showScores: true }
                );
                
                // 显示可视化
                PlantVis.displayVisualization(visContainer, canvas, {
                    allowThresholdChange: true,
                    initialThreshold: 0.5,
                    enableModeSwitch: true,
                    showControls: true,
                    allowDownload: true
                });
            }
        }
    });
});