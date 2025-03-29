/**
 * 植物病害检测可视化工具
 * 提供病害区域可视化和交互式结果分析功能
 * 优化版：增强模块化设计，减少事件监听器数量，增强热图渲染功能
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
            highlight: 'rgba(255, 255, 0, 0.4)', // 稍微提高高亮透明度
            boxDefault: '#2196F3',
            boxHighlight: '#FF4081'
        },
        opacity: 0.35, // 降低填充透明度，让底层图像更可见
        lineWidth: 3,   // 增加线宽以提高可见性
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
        let throttledThresholdUpdate = throttle(function(e) {
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
        }, 50); // 50ms的节流时间，平衡响应和性能

        document.body.addEventListener('input', throttledThresholdUpdate);
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
        
        // 创建画布 - 确保使用图像的原始尺寸
        const canvas = document.createElement('canvas');
        // 使用naturalWidth和naturalHeight获取原始尺寸
        const originalWidth = image instanceof HTMLImageElement ? image.naturalWidth : image.width;
        const originalHeight = image instanceof HTMLImageElement ? image.naturalHeight : image.height;
        canvas.width = originalWidth;
        canvas.height = originalHeight;
        
        const ctx = canvas.getContext('2d');
        
        // 绘制原始图像 - 确保使用原始尺寸
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        // 过滤分数低于阈值的检测结果
        const filteredDetections = detections.filter(det => det.score >= opts.scoreThreshold);
        
        // 调试输出
        console.log('绘制边界框，检测结果:', filteredDetections);
        console.log('画布尺寸:', canvas.width, 'x', canvas.height);
        
        // 绘制每个检测框
        filteredDetections.forEach((det, index) => {
            // 检查box格式，确保使用正确的坐标
            let x1, y1, x2, y2, width, height; // 确保所有变量都已提前声明
            
            if (Array.isArray(det.box)) {
                // 如果box是数组格式 - 确保正确理解格式
                if (det.box.length === 4) {
                    // 检查是否为[x1, y1, x2, y2]还是[x, y, width, height]
                    if (det.box[2] < image.width && det.box[3] < image.height) {
                        // 可能是[x, y, width, height]格式
                        x1 = det.box[0];
                        y1 = det.box[1];
                        width = det.box[2];
                        height = det.box[3];
                        x2 = x1 + width;
                        y2 = y1 + height;
                    } else {
                        // 假设是[x1, y1, x2, y2]格式
                        x1 = det.box[0];
                        y1 = det.box[1];
                        x2 = det.box[2];
                        y2 = det.box[3];
                    }
                } else {
                    console.warn('无效的边界框格式:', det.box);
                    x1 = 0; y1 = 0; x2 = 100; y2 = 100;
                }
            } else if (det.bbox) {
                // 从bbox对象中获取
                x1 = det.bbox.x || 0;
                y1 = det.bbox.y || 0;
                width = det.bbox.width || 0;
                height = det.bbox.height || 0;
                x2 = x1 + width;
                y2 = y1 + height;
            } else {
                // 默认值
                console.warn('未找到有效的边界框坐标，使用默认值');
                const defaultSize = Math.min(image.width, image.height) * 0.2;
                x1 = image.width/2 - defaultSize/2;
                y1 = image.height/2 - defaultSize/2;
                x2 = x1 + defaultSize;
                y2 = y1 + defaultSize;
            }
            
            // 确保坐标在图像范围内且正确计算宽高
            x1 = Math.max(0, Math.min(x1, image.width));
            y1 = Math.max(0, Math.min(y1, image.height));
            x2 = Math.max(0, Math.min(x2, image.width));
            y2 = Math.max(0, Math.min(y2, image.height));
            
            width = Math.max(1, x2 - x1);
            height = Math.max(1, y2 - y1);
            
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
                    
                    // 改进标签位置计算
                    const textBgHeight = textHeight + 6; // 增加高度提供垂直空间
                    
                    // 优先在框的上方放置标签，如果空间不足则放在框内顶部
                    let textY = y1 - 5; // 默认在框上方
                    
                    // 如果标签会超出图像顶部，则放在框的内部顶部
                    if (textY - textBgHeight < 0) {
                        textY = y1 + textBgHeight;
                    }
                    
                    // 计算背景矩形位置
                    const textX = x1;
                    const textBgY = textY - textBgHeight;
                    
                    // 绘制半透明深色背景，确保文本可读性
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
                    ctx.fillRect(textX, textBgY, textWidth + 6, textBgHeight);
                    
                    // 绘制标签文本，使用白色确保可读性
                    ctx.fillStyle = '#FFFFFF';
                    ctx.fillText(displayText, textX + 3, textY - 3);
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
            radius: options.radius || 25, // 稍微减小半径使热点更聚焦
            blur: options.blur || 25,     // 增加模糊以使过渡更平滑
            gradient: options.gradient || {
                0.1: 'blue',    // 起始于0.1而不是0.2
                0.3: 'cyan',
                0.5: 'lime',
                0.7: 'yellow',
                0.95: 'red'     // 使用0.95而不是1.0，让最高点更明显
            }
        };
        
        // 创建画布 - 确保使用图像的原始尺寸
        const canvas = document.createElement('canvas');
        const originalWidth = image instanceof HTMLImageElement ? image.naturalWidth : image.width;
        const originalHeight = image instanceof HTMLImageElement ? image.naturalHeight : image.height;
        
        canvas.width = originalWidth;
        canvas.height = originalHeight;
        
        const ctx = canvas.getContext('2d');
        
        // 绘制原始图像
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        console.log('开始渲染热图, heatmap.js可用:', !!window.h337);
        console.log('热图画布尺寸:', canvas.width, 'x', canvas.height);
        
        try {
            // 使用heatmap.js库生成热图
            if (window.h337) {
                // 创建临时容器
                const tempContainer = document.createElement('div');
                tempContainer.style.width = `${canvas.width}px`;
                tempContainer.style.height = `${canvas.height}px`;
                tempContainer.style.position = 'absolute';
                tempContainer.style.left = '-9999px';
                document.body.appendChild(tempContainer);
                
                // 验证热图数据的有效性
                const validData = heatmapData.filter(point => {
                    return point.x >= 0 && point.y >= 0 && 
                           point.x <= canvas.width && point.y <= canvas.height &&
                           point.radius > 0 && point.value >= 0;
                });
                
                // 配置热图
                const heatmapInstance = h337.create({
                    container: tempContainer,
                    radius: opts.radius,
                    maxOpacity: opts.opacity,
                    minOpacity: 0.2,  // 提高最小不透明度
                    blur: opts.blur,
                    gradient: opts.gradient
                });
                
                // 设置热图数据
                heatmapInstance.setData({
                    max: 1,
                    min: 0.1,  // 设置最小值，避免弱信号被抹除
                    data: validData
                });
                
                // 获取热图画布
                const heatCanvas = tempContainer.querySelector('canvas');
                
                // 将热图叠加到原始画布上
                ctx.globalAlpha = opts.opacity;
                ctx.drawImage(heatCanvas, 0, 0);
                ctx.globalAlpha = 1.0;
                
                // 移除临时容器
                document.body.removeChild(tempContainer);
            } else {
                console.warn('heatmap.js不可用，使用备用方法渲染');
                fallbackHeatmapRender(ctx, heatmapData, canvas.width, canvas.height);
            }
        } catch (error) {
            console.error('热图渲染出错:', error);
            // 使用备用方法渲染
            console.log('使用备用方法渲染热图');
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
        // 先创建一个单独的热图层
        const heatLayer = document.createElement('canvas');
        heatLayer.width = width;
        heatLayer.height = height;
        const heatCtx = heatLayer.getContext('2d');
        
        // 绘制每个热点
        heatmapData.forEach(point => {
            // 确保数据有效
            if (point.x < 0 || point.y < 0 || point.x > width || point.y > height) return;
            
            // 调整半径和透明度值
            const radius = Math.max(8, point.radius || 25);  // 确保最小半径
            const alpha = Math.min(0.95, Math.max(0.3, point.value || 0.5)); // 限制透明度范围
            
            try {
                // 创建径向渐变
                const gradient = heatCtx.createRadialGradient(
                    point.x, point.y, 0,
                    point.x, point.y, radius
                );
                
                // 使用更平滑、更符合热感知的颜色渐变
                gradient.addColorStop(0, `rgba(255, 0, 0, ${alpha})`);       // 红色(中心)
                gradient.addColorStop(0.3, `rgba(255, 120, 0, ${alpha * 0.8})`); // 橙色
                gradient.addColorStop(0.5, `rgba(255, 255, 0, ${alpha * 0.7})`); // 黄色
                gradient.addColorStop(0.75, `rgba(0, 255, 128, ${alpha * 0.4})`); // 绿色
                gradient.addColorStop(1, `rgba(0, 0, 255, 0)`);              // 透明蓝色(边缘)
                
                // 绘制圆
                heatCtx.beginPath();
                heatCtx.arc(point.x, point.y, radius, 0, Math.PI * 2, true);
                heatCtx.fillStyle = gradient;
                heatCtx.fill();
            } catch (e) {
                console.warn('创建渐变失败，使用简单圆形', e);
                // 简单圆形备用方案
                heatCtx.beginPath();
                heatCtx.arc(point.x, point.y, radius, 0, Math.PI * 2, true);
                heatCtx.fillStyle = `rgba(255, 0, 0, ${alpha * 0.7})`;
                heatCtx.fill();
            }
        });
        
        // 应用模糊效果使热图更平滑
        try {
            heatCtx.filter = 'blur(18px)'; // 增加模糊量
            heatCtx.drawImage(heatLayer, 0, 0);
        } catch (e) {
            console.warn('应用模糊效果失败', e);
        }
        
        // 将热图层叠加到主画布上，调整透明度
        ctx.globalAlpha = 0.65; // 略微调整透明度
        ctx.drawImage(heatLayer, 0, 0);
        ctx.globalAlpha = 1.0;
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
        
        // 选项整合
        const opts = {
            scoreThreshold: options.scoreThreshold || state.threshold,
            boxOpacity: options.boxOpacity || 0.65, // 降低框的不透明度
            heatmapOpacity: options.heatmapOpacity || 0.5 // 降低热图的不透明度
        };
        
        // 过滤检测结果
        const filteredDetections = detections.filter(det => det.score >= opts.scoreThreshold);
        
        // 创建最终画布 - 确保使用图像的原始尺寸
        const canvas = document.createElement('canvas');
        const originalWidth = image instanceof HTMLImageElement ? image.naturalWidth : image.width;
        const originalHeight = image instanceof HTMLImageElement ? image.naturalHeight : image.height;
        
        canvas.width = originalWidth;
        canvas.height = originalHeight;
        
        const ctx = canvas.getContext('2d');
        
        // 先绘制原始图像
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        console.log('混合模式画布尺寸:', canvas.width, 'x', canvas.height);
        
        try {
            // 生成热图层
            const heatmapData = detectionsToHeatmap(filteredDetections, image.width, image.height);
            const heatLayer = document.createElement('canvas');
            heatLayer.width = image.width;
            heatLayer.height = image.height;
            const heatCtx = heatLayer.getContext('2d');
            
            // 使用备用方法渲染热图 (更可靠)
            fallbackHeatmapRender(heatCtx, heatmapData, image.width, image.height);
            
            // 叠加热图层，使用调整后的透明度
            ctx.globalAlpha = opts.heatmapOpacity;
            ctx.drawImage(heatLayer, 0, 0);
            ctx.globalAlpha = 1.0;
            
            // 最后叠加边界框
            const boxOpts = {
                ...options, 
                opacity: opts.boxOpacity, 
                showLabels: true, 
                showScores: true
            };
            const boxCanvas = renderDetectionBoxes(image, filteredDetections, boxOpts);
            
            // 从边界框画布中只提取边界框部分 (忽略背景)
            ctx.drawImage(boxCanvas, 0, 0);
        } catch (error) {
            console.error('混合模式渲染出错：', error);
            // 出错时至少保证显示边界框
            const boxCanvas = renderDetectionBoxes(image, filteredDetections, options);
            ctx.drawImage(boxCanvas, 0, 0);
        }
        
        // 保存渲染结果
        state.lastRenderedCanvas = canvas;
        
        return canvas;
    }

    /**
     * 创建类GradCAM效果热图 (简化版)
     */
    function createGradCAMLikeHeatmap(image, detections, options = {}) {
        // 默认选项
        const opts = {
            opacity: options.opacity || 0.7,
            threshold: options.threshold || 0.3
        };
        
        // 创建画布 - 确保使用图像的原始尺寸
        const canvas = document.createElement('canvas');
        const originalWidth = image instanceof HTMLImageElement ? image.naturalWidth : image.width;
        const originalHeight = image instanceof HTMLImageElement ? image.naturalHeight : image.height;
        
        canvas.width = originalWidth;
        canvas.height = originalHeight;
        
        const ctx = canvas.getContext('2d');
        
        // 绘制原始图像
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        
        // 过滤检测结果
        const filteredDetections = detections.filter(det => det.score >= opts.threshold);
        
        console.log('GradCAM画布尺寸:', canvas.width, 'x', canvas.height);
        
        // 其余代码保持不变...
        
        // 创建热图层
        const heatLayer = document.createElement('canvas');
        heatLayer.width = image.width;
        heatLayer.height = image.height;
        const heatCtx = heatLayer.getContext('2d');
        
        // 为每个检测创建热区
        filteredDetections.forEach(det => {
            let box;
            if (Array.isArray(det.box)) {
                box = det.box; // [x1, y1, x2, y2]
            } else if (det.bbox) {
                const bbox = det.bbox;
                box = [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height];
            } else {
                return; // 跳过无效检测
            }
            
            // 确保坐标有效
            const x1 = Math.max(0, Math.min(box[0], canvas.width));
            const y1 = Math.max(0, Math.min(box[1], canvas.height));
            const x2 = Math.max(0, Math.min(box[2], canvas.width));
            const y2 = Math.max(0, Math.min(box[3], canvas.height));
            
            const width = Math.max(1, x2 - x1);
            const height = Math.max(1, y2 - y1);
            
            // 创建类GradCAM热区 (中心热度较高，边缘较低)
            const centerX = x1 + width / 2;
            const centerY = y1 + height / 2;
            const radius = Math.max(width, height) * 0.75;
            
            try {
                // 创建径向渐变
                const gradient = heatCtx.createRadialGradient(
                    centerX, centerY, 0,
                    centerX, centerY, radius
                );
                
                // 根据置信度调整透明度
                const alpha = det.score * opts.opacity;
                
                // GradCAM常用的红-黄配色，使用更平滑的渐变
                gradient.addColorStop(0, `rgba(255, 0, 0, ${alpha})`);
                gradient.addColorStop(0.4, `rgba(255, 128, 0, ${alpha * 0.9})`); // 新增中间橙色调
                gradient.addColorStop(0.7, `rgba(255, 255, 0, ${alpha * 0.7})`);
                gradient.addColorStop(0.9, `rgba(255, 255, 128, ${alpha * 0.5})`); // 新增浅黄色调
                gradient.addColorStop(1, `rgba(255, 255, 220, 0)`); // 淡黄色渐隐
                
                // 填充热区
                heatCtx.fillStyle = gradient;
                heatCtx.fillRect(x1, y1, width, height);
            } catch (e) {
                console.warn('创建GradCAM渐变失败：', e);
                // 简单备用方案
                heatCtx.fillStyle = `rgba(255, 0, 0, ${det.score * 0.6})`;
                heatCtx.fillRect(x1, y1, width, height);
            }
        });
        
        // 应用模糊效果
        try {
            heatCtx.filter = 'blur(10px)';
            heatCtx.drawImage(heatLayer, 0, 0);
        } catch (e) {
            console.warn('应用模糊效果失败', e);
        }
        
        // 叠加热图层到原始图像
        ctx.globalAlpha = opts.opacity;
        ctx.drawImage(heatLayer, 0, 0);
        ctx.globalAlpha = 1.0;
        
        return canvas;
    }

    /**
     * 在指定容器中显示可视化内容
     */
    function displayVisualization(container, canvas, options = {}) {
        if (!container || !canvas) return;
        
        // 清空容器
        container.innerHTML = '';
        
        // 创建一个包装器并添加滚动阴影效果
        const canvasWrapper = document.createElement('div');
        canvasWrapper.style.position = 'relative';
        canvasWrapper.style.width = '100%';
        canvasWrapper.style.overflow = 'hidden';
        canvasWrapper.style.borderRadius = '8px';
        canvasWrapper.style.boxShadow = '0 3px 10px rgba(0,0,0,0.15)';
        canvasWrapper.style.marginBottom = '15px';
        
        // 添加画布和设置样式 - 使用CSS控制显示尺寸
        canvas.style.maxWidth = '100%';
        canvas.style.height = 'auto';
        canvas.style.display = 'block';
        
        canvasWrapper.appendChild(canvas);
        container.appendChild(canvasWrapper);
        
        // 记录画布实际渲染尺寸和显示尺寸
        console.log('显示可视化：画布实际尺寸', canvas.width, 'x', canvas.height);
        
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
            downloadBtn.style.backgroundColor = '#4CAF50';
            downloadBtn.style.color = 'white';
            downloadBtn.style.border = 'none';
            downloadBtn.style.padding = '8px 16px';
            downloadBtn.style.borderRadius = '4px';
            downloadBtn.style.cursor = 'pointer';
            downloadBtn.style.display = 'flex';
            downloadBtn.style.alignItems = 'center';
            downloadBtn.style.gap = '6px';
            downloadBtn.style.margin = '0 auto 15px';
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
                    { label: '边界框', value: 'boxes', icon: 'fa-border-all' },
                    { label: '热图', value: 'heatmap', icon: 'fa-fire' },
                    { label: '混合', value: 'blend', icon: 'fa-object-group' },
                    { label: '热力图分析', value: 'gradcam', icon: 'fa-burn' }
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
            
            // 创建包含标签和值的容器
            const labelContainer = document.createElement('div');
            labelContainer.className = 'threshold-label-container';
            labelContainer.style.display = 'flex';
            labelContainer.style.justifyContent = 'space-between';
            labelContainer.style.marginBottom = '5px';
            
            const label = document.createElement('label');
            label.textContent = '置信度阈值: ';
            label.style.fontWeight = 'bold';
            
            const value = document.createElement('span');
            value.className = 'threshold-value';
            value.textContent = (options.threshold || config.defaultScoreThreshold).toFixed(2);
            value.style.fontWeight = 'bold';
            value.style.color = '#2196F3';
            
            // 添加标签和值到容器
            labelContainer.appendChild(label);
            labelContainer.appendChild(value);
            thresholdControl.appendChild(labelContainer);
            
            // 创建并设置滑块
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.className = 'threshold-slider';
            slider.min = '0';
            slider.max = '1';
            slider.step = '0.01';
            slider.value = options.threshold || config.defaultScoreThreshold;
            slider.style.width = '100%';
            slider.style.height = '6px';
            slider.style.borderRadius = '3px';
            slider.style.backgroundColor = '#e0e0e0';
            slider.style.outline = 'none';
            
            thresholdControl.appendChild(slider);
            controlPanel.appendChild(thresholdControl);
        }
        
        // 添加可视化模式选择
        if (options.enableModeSwitch) {
            const modeControl = document.createElement('div');
            modeControl.className = 'mode-control';
            modeControl.style.marginTop = '15px';
            
            const modeLabel = document.createElement('div');
            modeLabel.textContent = '可视化模式:';
            modeLabel.style.fontWeight = 'bold';
            modeLabel.style.marginBottom = '5px';
            modeControl.appendChild(modeLabel);
            
            // 获取可用模式
            const modeOptions = options.visModes || [
                { label: '边界框', value: 'boxes', icon: 'fa-border-all' },
                { label: '热图', value: 'heatmap', icon: 'fa-fire' },
                { label: '混合', value: 'blend', icon: 'fa-object-group' },
                { label: 'GradCAM', value: 'gradcam', icon: 'fa-burn' }
            ];
            
            const modeButtons = document.createElement('div');
            modeButtons.className = 'mode-buttons';
            modeButtons.style.display = 'flex';
            modeButtons.style.gap = '8px';
            modeButtons.style.flexWrap = 'wrap';
            
            // 创建每个模式的按钮
            modeOptions.forEach(mode => {
                const btn = document.createElement('button');
                btn.className = `vis-mode-switch ${mode.value === (options.currentMode || 'boxes') ? 'active' : ''}`;
                btn.dataset.mode = mode.value;
                
                // 使用提供的图标或默认图标
                const icon = mode.icon || 'fa-image';
                btn.innerHTML = `<i class="fas ${icon}"></i> ${mode.label}`;
                
                // 添加按钮样式
                btn.style.padding = '8px 12px';
                btn.style.border = '1px solid #ccc';
                btn.style.borderRadius = '4px';
                btn.style.backgroundColor = mode.value === (options.currentMode || 'boxes') ? '#2196F3' : '#f5f5f5';
                btn.style.color = mode.value === (options.currentMode || 'boxes') ? 'white' : '#333';
                btn.style.cursor = 'pointer';
                btn.style.transition = 'all 0.2s ease';
                btn.style.display = 'flex';
                btn.style.alignItems = 'center';
                btn.style.justifyContent = 'center';
                btn.style.gap = '5px';
                
                // 添加工具提示
                btn.title = `切换到${mode.label}模式`;
                
                modeButtons.appendChild(btn);
            });
            
            modeControl.appendChild(modeButtons);
            controlPanel.appendChild(modeControl);
        }
        
        // 整体控制面板样式
        controlPanel.style.backgroundColor = '#f9f9f9';
        controlPanel.style.borderRadius = '6px';
        controlPanel.style.padding = '12px';
        controlPanel.style.marginTop = '10px';
        controlPanel.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
        
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
     * @param {string} mode - 可视化模式 ('boxes', 'heatmap', 'blend', 'gradcam')
     */
    function changeVisualizationMode(mode) {
        if (!['boxes', 'heatmap', 'blend', 'gradcam'].includes(mode)) {
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
     */
    function detectionsToHeatmap(detections, imageWidth, imageHeight) {
        const heatmapData = [];
        console.log('转换热图数据，图像尺寸:', imageWidth, 'x', imageHeight);
        console.log(detections);
        
        // 确保检测结果不为空
        if (!detections || detections.length === 0) {
            return heatmapData;
        }
        
        // 图像尺寸检查
        imageWidth = Math.max(1, imageWidth || 100);
        imageHeight = Math.max(1, imageHeight || 100);
        
        // 更好的热图基础大小计算
        const diagonalLength = Math.sqrt(imageWidth * imageWidth + imageHeight * imageHeight);
        const baseRadius = diagonalLength * 0.05; // 更小的基础半径
        
        detections.forEach(det => {
            // 获取边界框 - 使用更稳健的处理方式
            let box;
            let x1, y1, x2, y2, width, height;
            
            if (Array.isArray(det.box)) {
                box = det.box;
                // 检测是[x1,y1,x2,y2]还是[x,y,width,height]格式
                if (det.box[2] < imageWidth && det.box[3] < imageHeight) {
                    // 可能是[x,y,width,height]
                    [x1, y1, width, height] = det.box;
                    x2 = x1 + width;
                    y2 = y1 + height;
                } else {
                    // 假定是[x1,y1,x2,y2]
                    [x1, y1, x2, y2] = det.box;
                    width = x2 - x1;
                    height = y2 - y1;
                }
            } else if (det.bbox) {
                x1 = det.bbox.x || 0;
                y1 = det.bbox.y || 0;
                width = det.bbox.width || 0;
                height = det.bbox.height || 0;
                x2 = x1 + width;
                y2 = y1 + height;
                box = [x1, y1, x2, y2];
            } else {
                // 如果没有边界框数据，使用整个图像
                x1 = 0;
                y1 = 0;
                x2 = imageWidth;
                y2 = imageHeight;
                width = imageWidth;
                height = imageHeight;
                box = [x1, y1, x2, y2];
            }
            
            // 确保坐标有效
            x1 = Math.max(0, Math.min(x1, imageWidth));
            y1 = Math.max(0, Math.min(y1, imageHeight));
            x2 = Math.max(0, Math.min(x2, imageWidth));
            y2 = Math.max(0, Math.min(y2, imageHeight));
            width = Math.max(1, x2 - x1);
            height = Math.max(1, y2 - y1);
            
            // 计算中心点
            const centerX = (x1 + x2) / 2;
            const centerY = (y1 + y2) / 2;
            
            // 添加主中心点 - 更精确的配置
            const mainPointRadius = Math.max(width, height) * 0.35; // 减小中心热点半径
            heatmapData.push({
                x: centerX,
                y: centerY,
                value: det.score || 0.5,
                radius: mainPointRadius
            });
            
            // 根据置信度和区域大小添加更多点
            const confidence = det.score || 0.5;
            // 根据置信度和区域大小调整点数
            const pointCount = 3 + Math.floor(confidence * 12); // 减少基础点，增加根据置信度变化的点
            const maxRadius = Math.min(width, height) * 0.25; // 减小随机点半径
            
            // 在区域内添加随机点，形成更自然的热图
            for (let i = 0; i < pointCount; i++) {
                // 使用高斯分布生成更自然的散布
                const u = Math.random() * 2 - 1; // -1 到 1
                const v = Math.random() * 2 - 1; // -1 到 1
                const distance = Math.sqrt(u*u + v*v) * 0.6; // 压缩标准差，使点更集中
                
                if (distance > 1) continue; // 丢弃超出单位圆的点
                
                // 将点映射到边界框内
                const offsetX = u * width * 0.35; // 使点更集中在中心区域
                const offsetY = v * height * 0.35;
                
                const pointX = Math.max(0, Math.min(centerX + offsetX, imageWidth));
                const pointY = Math.max(0, Math.min(centerY + offsetY, imageHeight));
                
                // 中心附近点的值更高，与中心的距离越远值越低
                const distanceFromCenter = Math.sqrt(offsetX*offsetX + offsetY*offsetY) / Math.sqrt(width*width/4 + height*height/4);
                const pointValue = confidence * (1 - distanceFromCenter*0.7); // 使值的衰减更快
                const pointRadius = maxRadius * (1 - distanceFromCenter*0.4); // 使半径的衰减更慢
                
                heatmapData.push({
                    x: pointX,
                    y: pointY,
                    value: pointValue,
                    radius: pointRadius
                });
            }
        });
        
        return heatmapData;
    }

    // 添加节流功能，用于提高滑动条操作时的渲染性能
    function throttle(func, delay) {
        let lastCall = 0;
        return function(...args) {
            const now = new Date().getTime();
            if (now - lastCall >= delay) {
                lastCall = now;
                return func.apply(this, args);
            }
        };
    }

    // 公开API
    return {
        // 基础方法
        initialize,
        
        // 渲染方法
        renderDetectionBoxes,
        renderHeatmap,
        renderBlendMode,
        createGradCAMLikeHeatmap,  // 添加新方法
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