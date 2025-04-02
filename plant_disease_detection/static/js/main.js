/**
 * 植物种类识别与病害检测系统前端交互脚本
 */
(function() {
    // 全局配置和状态 - 私有，不再暴露到全局作用域
    const APP_CONFIG = {
        // API端点
        apiEndpoints: {
            classification: '/api/classify',
            detection: '/api/detect',
            healthCheck: '/api/health'
        },
        // 上传配置
        upload: {
            maxSize: 10 * 1024 * 1024, // 10MB
            acceptedTypes: ['image/jpeg', 'image/png', 'image/jpg']
        }
    };

    // 应用状态 - 私有，不再暴露到全局作用域
    let appState = {
        currentTask: 'classification', // 'classification' 或 'detection'
        isProcessing: false,
        uploadedImage: null,
        results: null,
        detectedPlantType: null,  // 已有，保存检测到的植物类型
        plantIdentified: false,   // 新增，标记是否已完成植物识别
        detectionReady: false     // 新增，标记是否可以进行病害检测
    };

    /**
     * 初始化应用
     */
    function initializeApp() {
        console.log('初始化植物识别与病害检测应用...');
        
        // 检查API服务可用性
        checkApiHealth();
        
        // 设置事件监听器
        setupEventListeners();
        
        // 初始化UI组件
        initializeUIComponents();
    }

    /**
     * 设置事件监听器
     */
    function setupEventListeners() {
        // 上传区域的拖放事件
        const dropZone = document.getElementById('drop-zone');
        if (dropZone) {
            dropZone.addEventListener('dragover', handleDragOver);
            dropZone.addEventListener('dragleave', handleDragLeave);
            dropZone.addEventListener('drop', handleDrop);
            dropZone.addEventListener('click', triggerFileInput);
        }
        
        // 文件输入框变更事件
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', handleFileSelect);
        }
        
        // 任务选择切换
        const taskSwitches = document.querySelectorAll('.task-switch');
        taskSwitches.forEach(el => {
            el.addEventListener('click', handleTaskSwitch);
        });
        
        // 提交按钮点击事件
        const submitButton = document.getElementById('submit-button');
        if (submitButton) {
            submitButton.addEventListener('click', handleSubmit);
        }
        
        // 结果清除按钮
        const clearButton = document.getElementById('clear-button');
        if (clearButton) {
            clearButton.addEventListener('click', clearResults);
        }
        
        // 结果导出按钮
        const exportButton = document.getElementById('export-button');
        if (exportButton) {
            exportButton.addEventListener('click', exportResults);
        }
    }

    /**
     * 初始化UI组件
     */
    function initializeUIComponents() {
        // 初始化结果区域
        const resultsContainer = document.getElementById('results-container');
        if (resultsContainer) {
            resultsContainer.style.display = 'none';
        }
        
        // 初始化提示信息
        updateStatusMessage('请上传图像以进行植物识别或病害检测');
        
        // 初始化任务切换按钮状态
        updateTaskSwitchUI();

        // 新增: 初始化可视化相关DOM元素
        if (resultsContainer) {
            // 如果不存在，创建可视化容器
            if (!document.getElementById('visualization-container')) {
                const visContainer = document.createElement('div');
                visContainer.id = 'visualization-container';
                visContainer.className = 'visualization-container';
                visContainer.style.display = 'none'; // 初始隐藏
                resultsContainer.appendChild(visContainer);
            }
        }
        
        // 监听可视化组件事件
        document.addEventListener('visualization-threshold-update', function(e) {
            if (appState.currentTask === 'detection' && appState.results) {
                // 根据新阈值更新可视化
                updateVisualizationWithThreshold(e.detail.threshold);
            }
        });
        
        document.addEventListener('mode-change', function(e) {
            if (appState.currentTask === 'detection' && appState.results) {
                // 切换可视化模式
                updateVisualizationMode(e.detail.mode);
            }
        });
    }

    /**
     * 检查API服务健康状态
     */
    function checkApiHealth() {
        fetch(APP_CONFIG.apiEndpoints.healthCheck)
            .then(response => {
                if (response.ok) {
                    console.log('API服务正常运行');
                    updateStatusMessage('系统就绪，请上传图像');
                } else {
                    console.error('API服务不可用');
                    updateStatusMessage('系统服务不可用，请稍后再试', 'error');
                }
            })
            .catch(error => {
                console.error('API健康检查失败:', error);
                updateStatusMessage('无法连接到系统服务', 'error');
            });
    }

    /**
     * 当拖动文件经过时的处理
     */
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // 添加活跃状态样式
        document.getElementById('drop-zone').classList.add('active');
    }

    /**
     * 当拖动离开时的处理
     */
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // 移除活跃状态样式
        document.getElementById('drop-zone').classList.remove('active');
    }

    /**
     * 当文件被拖放到区域时的处理
     */
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // 移除活跃状态样式
        document.getElementById('drop-zone').classList.remove('active');
        
        // 获取文件
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageFile(files[0]);
        }
    }

    /**
     * 打开文件选择对话框
     */
    function triggerFileInput() {
        document.getElementById('file-input').click();
    }

    /**
     * 处理文件选择事件
     */
    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            handleImageFile(files[0]);
        }
    }

    /**
     * 处理上传的图片文件
     */
    function handleImageFile(file) {
        // 验证文件类型
        if (!APP_CONFIG.upload.acceptedTypes.includes(file.type)) {
            updateStatusMessage('不支持的文件格式。请上传JPG或PNG图像', 'error');
            return;
        }
        
        // 验证文件大小
        if (file.size > APP_CONFIG.upload.maxSize) {
            updateStatusMessage(`文件过大。最大允许${APP_CONFIG.upload.maxSize/1024/1024}MB`, 'error');
            return;
        }
        
        // 显示预览
        displayImagePreview(file);
        
        // 更新应用状态
        appState.uploadedImage = file;
        appState.results = null;
        
        // 更新UI
        updateStatusMessage('图像已准备好，可以开始处理');
        document.getElementById('submit-button').disabled = false;
        
        // 隐藏之前的结果
        const resultsContainer = document.getElementById('results-container');
        if (resultsContainer) resultsContainer.style.display = 'none';
    }

    /**
     * 显示上传图片的预览
     */
    function displayImagePreview(file) {
        const reader = new FileReader();
        const previewContainer = document.getElementById('preview-container');
        
        reader.onload = function(e) {
            previewContainer.innerHTML = `
                <div class="preview-image-container">
                    <img src="${e.target.result}" alt="预览图像" class="preview-image">
                </div>
            `;
            previewContainer.style.display = 'block';
        };
        
        reader.readAsDataURL(file);
    }

    /**
     * 处理任务类型切换
     */
    function handleTaskSwitch(e) {
        const taskType = e.target.dataset.task;
        if (taskType && taskType !== appState.currentTask) {
            // 更新应用状态
            appState.currentTask = taskType;
            updateTaskSwitchUI();
            
            // 如果是切换到分类任务
            if (taskType === 'classification') {
                updateStatusMessage('切换到植物识别模式');
            } 
            // 如果是切换到病害检测任务
            else if (taskType === 'detection') {
                // 检查是否已有识别的植物类型
                if (appState.plantIdentified && appState.detectedPlantType) {
                    updateStatusMessage(`已切换到病害检测模式，将检测${appState.detectedPlantType}的病害`);
                    
                    // 如果有上传的图像，则启用提交按钮
                    if (appState.uploadedImage) {
                        document.getElementById('submit-button').disabled = false;
                    }
                } else {
                    updateStatusMessage('请先完成植物识别，再进行病害检测', 'warning');
                }
            }
        }
    }

    /**
     * 更新任务切换UI
     */
    function updateTaskSwitchUI() {
        document.querySelectorAll('.task-switch').forEach(el => {
            if (el.dataset.task === appState.currentTask) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });
    }

    /**
     * 处理提交按钮点击
     */
    function handleSubmit() {
        if (!appState.uploadedImage || appState.isProcessing) return;
        
        // 如果是检测任务，但没有植物类型信息
        if (appState.currentTask === 'detection' && !appState.detectedPlantType) {
            updateStatusMessage('请先进行植物识别，再进行病害检测', 'warning');
            
            // 自动切换到分类任务
            appState.currentTask = 'classification';
            updateTaskSwitchUI();
            return;
        }
        
        // 更新状态
        appState.isProcessing = true;
        updateStatusMessage('正在处理图像，请稍候...');
        document.getElementById('submit-button').disabled = true;
        
        // 显示加载指示器
        showLoadingIndicator();
        
        // 准备表单数据
        const formData = new FormData();
        formData.append('image', appState.uploadedImage);
        
        // 如果是检测任务且已知植物类型，则添加植物类型
        if (appState.currentTask === 'detection' && appState.detectedPlantType) {
            formData.append('plant_type', appState.detectedPlantType);
            console.log(`检测请求附加植物类型: ${appState.detectedPlantType}`);
        }
        
        // 确定API端点
        const endpoint = appState.currentTask === 'classification' 
            ? APP_CONFIG.apiEndpoints.classification 
            : APP_CONFIG.apiEndpoints.detection;
        
        // 发送请求
        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // 隐藏加载指示器
            hideLoadingIndicator();
            
            // 更新应用状态
            appState.isProcessing = false;
            appState.results = data;
            
            // 如果是分类任务，存储植物类型信息
            if (appState.currentTask === 'classification' && data.predictions && data.predictions.length > 0) {
                // 使用提取的纯植物类型
                appState.detectedPlantType = extractPlantType(data.predictions[0].class_name);
                appState.plantIdentified = true;
                appState.detectionReady = true;
                console.log(`植物识别成功: ${appState.detectedPlantType}`);
            }
            
            // 显示处理结果
            displayResults(data);
            
            // 更新状态消息
            updateStatusMessage('处理完成');
        })
        .catch(error => {
            console.error('图像处理出错:', error);
            hideLoadingIndicator();
            appState.isProcessing = false;
            updateStatusMessage('图像处理失败，请重试', 'error');
        });
    }

    /**
     * 显示处理结果
     */
    function displayResults(results) {
        const resultsContainer = document.getElementById('results-container');
        if (!resultsContainer) return;
        
        // 清除之前的结果
        resultsContainer.innerHTML = '';
        
        // 创建结果标题
        const resultTitle = document.createElement('h3');
        resultTitle.textContent = appState.currentTask === 'classification' ? '植物识别结果' : '病害检测结果';
        resultsContainer.appendChild(resultTitle);
        
        // 根据任务类型显示不同结果
        if (appState.currentTask === 'classification') {
            displayClassificationResults(results, resultsContainer);
        } else {
            displayDetectionResults(results, resultsContainer);
        }
        
        // 添加操作按钮
        addResultActions(resultsContainer);
        
        // 显示结果容器
        resultsContainer.style.display = 'block';
        
        // 平滑滚动到结果区域
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * 显示分类结果
     */
    function displayClassificationResults(results, container) {
        // 创建分类结果卡片
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card classification-result';
        
        // 获取主要预测结果
        const topPrediction = results.predictions[0];
        
        // 提取纯植物名称（去除可能的病害部分）
        let plantType = topPrediction.class_name;
        if (plantType.includes('-')) {
            plantType = plantType.split('-')[0].trim();
        }
        
        // 获取植物描述
        const plantDescription = results.plant_info && results.plant_info.description ? 
                               results.plant_info.description : '无植物描述';
        
        // 显示植物名称和置信度
        resultCard.innerHTML = `
            <div class="result-header">
                <h4>${plantType}</h4>
                <div class="confidence">
                    <span class="confidence-value">${(topPrediction.confidence * 100).toFixed(2)}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${topPrediction.confidence * 100}%"></div>
                    </div>
                </div>
            </div>
            <div class="result-details">
                <div class="plant-info">
                    <p class="plant-description">${plantDescription}</p>
                    
                    ${results.plant_info && results.plant_info.general_care && results.plant_info.general_care.length > 0 ? `
                    <div class="general-care">
                        <h5>植物护理要点：</h5>
                        <ul class="care-list">
                            ${results.plant_info.general_care.map(tip => `<li>${tip}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                </div>
                
                <div class="other-predictions">
                    <h5>其他可能的植物种类：</h5>
                    <ul class="predictions-list">
                        ${results.predictions.slice(1, 4).map(pred => {
                            // 提取纯植物名称
                            let predPlantType = pred.class_name;
                            if (predPlantType.includes('-')) {
                                predPlantType = predPlantType.split('-')[0].trim();
                            }
                            return `
                            <li>
                                <span class="prediction-name">${predPlantType}</span>
                                <span class="prediction-confidence">${(pred.confidence * 100).toFixed(2)}%</span>
                            </li>
                            `;
                        }).join('')}
                    </ul>
                </div>
                
                <div class="actions action-buttons-container">
                    <button class="action-button secondary" onclick="showPlantInfo('${plantType}')">
                        <i class="fas fa-info-circle"></i> 查看植物详细信息
                    </button>
                    <button class="action-button primary" onclick="switchToDetection('${plantType}')">
                        <i class="fas fa-search-plus"></i> 检测可能的病害
                    </button>
                </div>
            </div>
        `;
        
        // 添加到容器
        container.appendChild(resultCard);
        
        // 更新应用状态中保存的植物类型
        appState.detectedPlantType = plantType;
        appState.plantIdentified = true;
        appState.detectionReady = true;
    }

    // 修改或添加 extractPlantType 函数，确保正确提取纯植物名称

    function extractPlantType(className) {
        // 处理中文格式 "植物-病害"
        if (className.includes('-')) {
            return className.split('-')[0].trim();
        }
        
        // 处理英文格式 "Plant___Disease"
        if (className.includes('___')) {
            const plant = className.split('___')[0].trim();
            
            // 将英文植物名映射到中文
            const plantMappings = {
                'Apple': '苹果',
                'Blueberry': '蓝莓',
                'Cherry': '樱桃',
                'Corn': '玉米',
                'Grape': '葡萄',
                'Orange': '橙子',
                'Peach': '桃子',
                'Pepper': '甜椒',
                'Potato': '土豆',
                'Raspberry': '树莓',
                'Soybean': '大豆',
                'Squash': '西葫芦',
                'Strawberry': '草莓',
                'Tomato': '番茄'
            };
            
            return plantMappings[plant] || plant;
        }
        
        // 对于没有分隔符的情况，直接返回原值
        return className;
    }

    // 新增：添加显示植物详细信息的函数(在extractPlantType后添加)
    function showPlantInfo(plantType) {
        // 创建对话框
        const modal = document.getElementById('treatment-modal');
        const modalBody = document.getElementById('modal-body');
        
        // 显示加载中
        modalBody.innerHTML = '<div class="loading">加载中...</div>';
        modal.style.display = 'block';
        
        // 获取植物详细信息
        fetch(`/api/plant_info?plant=${encodeURIComponent(plantType)}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    modalBody.innerHTML = `<div class="error">${data.error}</div>`;
                    return;
                }
                
                // 显示植物信息
                let htmlContent = `
                    <h3>${plantType} - 植物详情</h3>
                    <div class="plant-detail">
                        <p class="description">${data.description || '暂无描述'}</p>
                    </div>
                `;
                
                // 添加护理建议
                if (data.general_care && data.general_care.length > 0) {
                    htmlContent += `
                        <div class="care-tips">
                            <h4>护理建议</h4>
                            <ul>
                                ${data.general_care.map(tip => `<li>${tip}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }
                
                // 添加常见病害信息
                if (data.common_diseases && data.common_diseases.length > 0) {
                    htmlContent += `
                        <div class="common-diseases">
                            <h4>常见病害</h4>
                            <ul>
                                ${data.common_diseases.map(disease => `<li>${disease}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }
                
                modalBody.innerHTML = htmlContent;
            })
            .catch(error => {
                console.error('获取植物信息失败:', error);
                modalBody.innerHTML = `<div class="error">获取植物信息失败</div>`;
            });
    }

    /**
     * 显示检测结果
     */
    function displayDetectionResults(results, container) {
        // 清除之前的结果
        container.innerHTML = '';

        // 创建结果标题
        const resultTitle = document.createElement('h3');
        resultTitle.textContent = '病害识别结果'; // 更改标题，更符合分类任务
        container.appendChild(resultTitle);

        // --- 修改: 结果卡片内容 ---
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card detection-as-classification-result'; // 可以用新类名

        // 从 results 中提取关键分类信息 (基于 routes.py 返回的新格式)
        const className = results.class_name || "无法识别";
        const confidence = results.confidence || 0.0;
        const plantType = results.plant_type || "未知植物";
        const diseaseName = results.disease_name || "未知病害";
        const treatmentInfo = results.treatment_info || {};
        const labelId = results.top_prediction ? results.top_prediction.label_id : -1; // 获取类别 ID

        let severityClass = 'unknown'; // 可以根据置信度估算
        if (confidence > 0.8) severityClass = 'severe';
        else if (confidence > 0.6) severityClass = 'moderate';
        else if (confidence > 0.4) severityClass = 'mild';

        resultCard.innerHTML = `
            <div class="result-header">
                <h4>${className}</h4>
                <div class="confidence">
                    <span class="confidence-value">${(confidence * 100).toFixed(2)}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
                    </div>
                </div>
                <span class="severity-badge ${severityClass}">置信度评估: ${severityClass}</span>
            </div>
            <div class="result-details">
                <p>识别来源: 检测器模型 (图像级检测)</p>
                ${Object.keys(treatmentInfo).length > 0 && !treatmentInfo.error ? `
                    <div class="treatment-summary">
                        <h5>初步治疗建议:</h5>
                        <p>${treatmentInfo.treatments ? treatmentInfo.treatments[0] : '暂无建议'}</p>
                        <button class="action-button secondary view-treatment-details" data-plant="${plantType}" data-disease="${diseaseName}">
                            <i class="fas fa-notes-medical"></i> 查看详细治疗方案
                        </button>
                    </div>
                ` : diseaseName.toLowerCase() !== '健康' ? `<p>未找到针对"${diseaseName}"的特定治疗建议。</p>` : '<p>植物看起来很健康！</p>'}

                <!-- 新增: Grad-CAM 相关按钮 -->
                ${labelId !== -1 ? `
                <div class="gradcam-actions">
                    <button class="action-button tertiary explain-button" data-label-id="${labelId}">
                        <i class="fas fa-eye"></i> 查看模型关注区域 (Grad-CAM)
                    </button>
                    <div class="gradcam-result-container" style="display: none; margin-top: 15px;">
                         <img src="" alt="Grad-CAM Visualization" class="gradcam-image" style="max-width: 100%; border: 1px solid #ddd;"/>
                         <p class="gradcam-loading" style="display: none;">正在生成解释...</p>
                         <p class="gradcam-error" style="color: red; display: none;"></p>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
        container.appendChild(resultCard);
        // --- 结束修改结果卡片 ---

        // --- 移除或注释掉旧的可视化调用 ---
        // const previewImage = document.querySelector('.preview-image');
        // if (previewImage && typeof PlantVis !== 'undefined') {
        //    // ... 旧的调用 PlantVis.renderDetectionBoxes 等代码 ...
        //    // PlantVis.displayVisualization(...);
        // }
        // --- 结束移除 ---

        // --- 添加新的事件监听 ---
        // 查看详细治疗方案按钮
        const treatmentBtn = resultCard.querySelector('.view-treatment-details');
        if (treatmentBtn) {
            treatmentBtn.addEventListener('click', function() {
                const plant = this.dataset.plant;
                const disease = this.dataset.disease;
                fetch(`/api/treatment?plant=${encodeURIComponent(plant)}&disease=${encodeURIComponent(disease)}`)
                    .then(res => res.json())
                    .then(treatmentData => {
                        // 假设你有一个函数 showDetailedTreatment 来显示模态框
                        showDetailedTreatment(treatmentData);
                    })
                    .catch(err => console.error('获取治疗信息失败:', err));
            });
        }

        // Grad-CAM 解释按钮
        const explainBtn = resultCard.querySelector('.explain-button');
        if (explainBtn) {
            explainBtn.addEventListener('click', handleExplainRequest); // 调用新的处理函数
        }
        // --- 结束添加 ---

        // 显示结果容器 (保留)
        container.style.display = 'block';
        container.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * 添加结果操作按钮
     */
    function addResultActions(container) {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'result-actions';
        
        actionsDiv.innerHTML = `
            <button id="export-button" class="action-button">
                <i class="fa fa-download"></i> 导出结果
            </button>
            <button id="clear-button" class="action-button">
                <i class="fa fa-trash"></i> 清除结果
            </button>
        `;
        
        container.appendChild(actionsDiv);
        
        // 添加事件监听
        document.getElementById('export-button').addEventListener('click', exportResults);
        document.getElementById('clear-button').addEventListener('click', clearResults);
    }

    /**
     * 导出结果
     */
    function exportResults() {
        if (!appState.results) return;
        
        // 准备要导出的数据
        const exportData = {
            taskType: appState.currentTask,
            timestamp: new Date().toISOString(),
            results: appState.results
        };
        
        // 创建JSON文件并下载
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const downloadLink = document.createElement('a');
        downloadLink.href = URL.createObjectURL(dataBlob);
        downloadLink.download = `plant-${appState.currentTask}-result-${new Date().getTime()}.json`;
        
        // 触发下载
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    }

    /**
     * 清除结果
     */
    function clearResults() {
        // 保留原有清除逻辑
        
        // 新增: 清除可视化内容
        const visContainer = document.getElementById('visualization-container');
        if (visContainer) {
            visContainer.innerHTML = '';
            // 如果visualization.js有清理方法，也应调用它
            if (typeof PlantVis !== 'undefined' && PlantVis.clearVisualizations) {
                PlantVis.clearVisualizations();
            }
        }
        
        // 重置应用状态
        appState.results = null;
        appState.detectedPlantType = null;  // 重置植物类型
        appState.plantIdentified = false;   // 重置植物识别状态
        appState.detectionReady = false;    // 重置检测准备状态
        
        // 隐藏结果容器
        const resultsContainer = document.getElementById('results-container');
        if (resultsContainer) resultsContainer.style.display = 'none';
        
        // 重置预览
        const previewContainer = document.getElementById('preview-container');
        if (previewContainer) previewContainer.style.display = 'none';
        
        // 重置文件输入
        const fileInput = document.getElementById('file-input');
        if (fileInput) fileInput.value = '';
        
        // 更新状态消息
        updateStatusMessage('请上传新图像以继续');
        
        // 禁用提交按钮
        document.getElementById('submit-button').disabled = true;
    }

    /**
     * 更新状态消息
     */
    function updateStatusMessage(message, type = 'info') {
        const statusElement = document.getElementById('status-message');
        if (!statusElement) return;
        
        // 设置消息和类型
        statusElement.textContent = message;
        
        // 清除现有类
        statusElement.className = '';
        
        // 添加基本类和类型特定类
        statusElement.classList.add('status-message');
        statusElement.classList.add(type); // 添加类型类名
        
        // 如果是警告或错误消息，使其暂时性高亮
        if (type === 'error' || type === 'warning') {
            statusElement.classList.add('highlight');
            setTimeout(() => {
                statusElement.classList.remove('highlight');
            }, 2000);
        }
    }

    /**
     * 显示加载指示器
     */
    function showLoadingIndicator() {
        let loader = document.getElementById('loading-indicator');
        
        if (!loader) {
            loader = document.createElement('div');
            loader.id = 'loading-indicator';
            loader.innerHTML = `
                <div class="spinner"></div>
                <p>处理中，请稍候...</p>
            `;
            document.body.appendChild(loader);
        }
        
        loader.style.display = 'flex';
    }

    /**
     * 隐藏加载指示器
     */
    function hideLoadingIndicator() {
        const loader = document.getElementById('loading-indicator');
        if (loader) {
            loader.style.display = 'none';
        }
    }

    /**
     * 更新可视化阈值
     */
    function updateVisualizationWithThreshold(threshold) {
        if (!appState.results || !appState.results.detections) return;
        
        const originalImage = document.querySelector('.preview-image');
        if (!originalImage) return;
        
        const visContainer = document.getElementById('visualization-container');
        if (!visContainer || typeof PlantVis === 'undefined') return;
        
        // 确保图像已完全加载，获取其原始尺寸
        if (!originalImage.complete) {
            originalImage.onload = function() {
                performVisualizationUpdate(originalImage, threshold);
            };
        } else {
            performVisualizationUpdate(originalImage, threshold);
        }
        
        function performVisualizationUpdate(image, threshold) {
            // 获取检测结果并格式化
            const detections = appState.results.detections;
            const formattedDetections = detections.map(det => ({
                box: [det.bbox.x, det.bbox.y, det.bbox.x + det.bbox.width, det.bbox.y + det.bbox.height],
                score: det.score,
                class_name: det.class_name,
                severity: det.severity
            }));
            
            // 根据当前模式选择渲染方法
            let canvas;
            const currentMode = document.querySelector('.vis-mode-switch.active')?.dataset.mode || 'boxes';
            
            switch(currentMode) {
                case 'heatmap':
                    const heatmapData = PlantVis.detectionsToHeatmap(
                        formattedDetections.filter(det => det.score >= threshold),
                        image.naturalWidth, // 使用原始宽度
                        image.naturalHeight // 使用原始高度
                    );
                    
                    // 添加保护，确保所有点的半径为正
                    if (heatmapData && heatmapData.length > 0) {
                        heatmapData.forEach(point => {
                            point.radius = Math.max(point.radius, 1);
                        });
                    }
                    
                    canvas = PlantVis.renderHeatmap(image, heatmapData);
                    break;
                    
                case 'blend':
                    canvas = PlantVis.renderBlendMode(image, formattedDetections, {
                        scoreThreshold: threshold
                    });
                    break;
                
                case 'gradcam':
                    canvas = PlantVis.createGradCAMLikeHeatmap(image, formattedDetections, {
                        opacity: 0.7,
                        threshold: threshold
                    });
                    break;
                    
                case 'boxes':
                default:
                    canvas = PlantVis.renderDetectionBoxes(image, formattedDetections, {
                        showLabels: true,
                        showScores: true,
                        scoreThreshold: threshold
                    });
                    break;
            }
            
            // 显示可视化
            PlantVis.displayVisualization(visContainer, canvas, {
                allowThresholdChange: true,
                initialThreshold: threshold,
                enableThreshold: true,
                enableModeSwitch: true,
                visModes: [
                    { label: '边界框', value: 'boxes' },
                    { label: '热图', value: 'heatmap' },
                    { label: '混合', value: 'blend' },
                    { label: 'GradCAM', value: 'gradcam' }
                ],
                currentMode: currentMode,
                showControls: true,
                allowDownload: true
            });
        }
    }

    /**
     * 更新可视化模式
     */
    function updateVisualizationMode(mode) {
        if (!appState.results || !appState.results.detections) return;
        
        const originalImage = document.querySelector('.preview-image');
        if (!originalImage) return;
        
        const visContainer = document.getElementById('visualization-container');
        if (!visContainer || typeof PlantVis === 'undefined') return;
        
        // 确保图像已完全加载
        if (!originalImage.complete) {
            originalImage.onload = function() {
                performVisualizationModeUpdate(originalImage, mode);
            };
        } else {
            performVisualizationModeUpdate(originalImage, mode);
        }
        
        function performVisualizationModeUpdate(image, mode) {
            console.log('切换可视化模式:', mode);
            
            // 获取检测结果并格式化
            const detections = appState.results.detections;
            const formattedDetections = detections.map(det => ({
                box: [det.bbox.x, det.bbox.y, det.bbox.x + det.bbox.width, det.bbox.y + det.bbox.height],
                score: det.score,
                class_name: det.class_name,
                severity: det.severity
            }));
            
            let canvas;
            let threshold = mode === 'boxes' ? 0.5 : 0.3; // 不同模式可使用不同默认阈值
            
            // 根据模式选择不同的渲染方法
            switch(mode) {
                case 'heatmap':
                    // 使用PlantVis提供的方法直接生成热图数据
                    const heatmapData = PlantVis.detectionsToHeatmap(
                        formattedDetections.filter(det => det.score >= threshold), 
                        image.naturalWidth, // 使用原始宽度
                        image.naturalHeight // 使用原始高度
                    );
                    canvas = PlantVis.renderHeatmap(image, heatmapData);
                    break;
                    
                case 'blend':
                    // 使用混合模式渲染
                    canvas = PlantVis.renderBlendMode(image, formattedDetections, {
                        scoreThreshold: threshold
                    });
                    break;
                    
                case 'gradcam':
                    canvas = PlantVis.createGradCAMLikeHeatmap(image, formattedDetections, {
                        opacity: 0.7,
                        threshold: threshold
                    });
                    break;
                    
                case 'boxes':
                default:
                    // 默认使用边界框模式
                    canvas = PlantVis.renderDetectionBoxes(image, formattedDetections, {
                        showLabels: true,
                        showScores: true,
                        scoreThreshold: threshold
                    });
                    break;
            }
            
            // 统一处理可视化显示
            PlantVis.displayVisualization(visContainer, canvas, {
                allowThresholdChange: true,
                initialThreshold: threshold,
                enableThreshold: true,
                enableModeSwitch: true,
                visModes: [
                    { label: '边界框', value: 'boxes' },
                    { label: '热图', value: 'heatmap' },
                    { label: '混合', value: 'blend' },
                    { label: 'GradCAM', value: 'gradcam' }
                ],
                currentMode: mode,
                showControls: true,
                allowDownload: true
            });
        }
    }

    /**
     * 根据检测结果生成热图数据
     */
    function generateHeatmapData(detections, width, height) {
        const heatmapData = [];
        detections.forEach(detection => {
            // 计算检测框中心点
            const centerX = detection.bbox.x + detection.bbox.width / 2;
            const centerY = detection.bbox.y + detection.bbox.height / 2;
            
            // 以检测框的大小为基础计算热点半径
            // 添加保护逻辑，确保半径永远为正数
            let radius = Math.max(detection.bbox.width, detection.bbox.height) * 0.7;
            radius = Math.max(radius, 1); // 确保半径至少为1像素
            
            heatmapData.push({
                x: centerX,
                y: centerY,
                value: detection.score,
                radius: radius
            });
        });
        return heatmapData;
    }

    // 修改 switchToDetection 函数，确保它正确处理纯植物名称

    /**
     * 切换到检测模式的辅助函数
     */
    function switchToDetection(plantType) {
        // 更新状态
        appState.currentTask = 'detection';
        appState.detectedPlantType = plantType; // 确保存储的是纯植物名称
        appState.plantIdentified = true;
        appState.detectionReady = true;
        
        // 更新UI以显示当前模式
        updateTaskSwitchUI();
        
        // 更新状态消息
        updateStatusMessage(`正在准备检测${plantType}的病害...`, 'info');
        
        // 自动进行提交，无需用户再次点击"开始分析"
        setTimeout(() => {
            handleSubmit();
        }, 500);
    }

    // 添加显示详细治疗信息的模态框函数
    function showDetailedTreatment(treatment) {
        // 获取模态框元素
        let modal = document.getElementById('treatment-modal');
        
        // 如果模态框不存在，创建一个
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'treatment-modal';
            modal.className = 'modal';
            
            const modalContent = document.createElement('div');
            modalContent.className = 'modal-content';
            
            const closeSpan = document.createElement('span');
            closeSpan.className = 'close';
            closeSpan.innerHTML = '&times;';
            closeSpan.onclick = function() {
                modal.style.display = 'none';
            };
            
            modalContent.appendChild(closeSpan);
            const modalBody = document.createElement('div');
            modalBody.id = 'modal-body';
            modalContent.appendChild(modalBody);
            
            modal.appendChild(modalContent);
            document.body.appendChild(modal);
            
            // 点击模态框外部时关闭
            window.onclick = function(event) {
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            };
        }
        
        // 准备模态框内容
        const modalBody = document.getElementById('modal-body');
        
        // 添加返回按钮和标题
        modalBody.innerHTML = `
            <div class="treatment-header-nav">
                <h3>${treatment.plant_name}的${treatment.disease_name}详细治疗方案</h3>
            </div>
            
            <div class="treatment-section">
                <h4>症状描述</h4>
                <ul>
                    ${treatment.symptoms ? treatment.symptoms.map(s => `<li>${s}</li>`).join('') : '<li>无症状描述</li>'}
                </ul>
            </div>
            
            <div class="treatment-section">
                <h4>病因分析</h4>
                <p>${treatment.causes ? treatment.causes.join('</p><p>') : '无病因分析'}</p>
            </div>
            
            <div class="treatment-section">
                <h4>治疗方法</h4>
                <ul>
                    ${treatment.treatments ? treatment.treatments.map(t => `<li>${t}</li>`).join('') : '<li>无治疗方法</li>'}
                </ul>
            </div>
            
            <div class="treatment-section">
                <h4>预防措施</h4>
                <ul>
                    ${treatment.prevention ? treatment.prevention.map(p => `<li>${p}</li>`).join('') : '<li>无预防措施</li>'}
                </ul>
            </div>
            
            ${treatment.organic_solutions ? `
            <div class="treatment-section organic">
                <h4>有机解决方案</h4>
                <ul>
                    ${treatment.organic_solutions.map(o => `<li>${o}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            ${treatment.severity ? `
            <div class="treatment-section">
                <h4>严重程度</h4>
                <p>${treatment.severity}</p>
            </div>
            ` : ''}
            
            <div class="treatment-actions">
                <button class="action-button secondary" id="close-treatment-modal">
                    <i class="fas fa-times"></i> 关闭
                </button>
            </div>
        `;
        
        // 显示模态框
        modal.style.display = 'block';
        
        // 添加关闭按钮事件
        document.getElementById('close-treatment-modal').addEventListener('click', function() {
            modal.style.display = 'none';
        });
    }

    // 添加处理治疗建议的辅助函数
    function processTreatmentRecommendations(treatments, container) {
        if (!treatments || treatments.length === 0) {
            console.log("没有治疗建议可处理");
            return;
        }
        
        console.log("处理治疗建议:", treatments);
        
        const recContainer = document.createElement('div');
        recContainer.className = 'treatment-recommendations';
        
        // 添加标题，使用更引人注目的样式
        const titleDiv = document.createElement('div');
        titleDiv.className = 'treatment-title';
        titleDiv.innerHTML = '<h4><i class="fas fa-leaf"></i> 智能治疗建议</h4>';
        titleDiv.style.borderBottom = "2px solid #4CAF50";
        titleDiv.style.paddingBottom = "8px";
        titleDiv.style.marginBottom = "10px";
        recContainer.appendChild(titleDiv);
        
        // 处理治疗建议内容
        const contentDiv = document.createElement('div');
        contentDiv.className = 'treatment-content';
        
        // 根据建议类型处理不同的显示方式
        treatments.forEach((treatment, index) => {
            const treatmentItem = document.createElement('div');
            treatmentItem.className = 'treatment-item';
            treatmentItem.style.marginBottom = "15px";
            
            if (typeof treatment === 'string') {
                // 简单字符串建议
                treatmentItem.innerHTML = `<p>${treatment}</p>`;
            } else if (typeof treatment === 'object') {
                // 结构化建议对象
                let urgency = '';
                if (treatment.urgency) {
                    const urgencyClass = 
                        treatment.urgency.includes('高') ? 'high-urgency' : 
                        treatment.urgency.includes('中') ? 'medium-urgency' : 'low-urgency';
                    urgency = `<span class="${urgencyClass}">${treatment.urgency}</span>`;
                }
                
                treatmentItem.innerHTML = `
                    <div class="treatment-header">
                        <h5>${treatment.disease || '检测到的病害'}</h5>
                        ${urgency}
                    </div>
                    <div class="treatment-content">
                        <p><strong>推荐方法:</strong> ${treatment.treatments ? treatment.treatments[0] : (treatment.description || treatment.method || '无详细说明')}</p>
                        ${treatment.treatments && treatment.treatments.length > 1 ? 
                            `<p class="more-treatments">+${treatment.treatments.length-1}种其他方法...</p>` : ''}
                    </div>
                `;
                
                // 添加点击事件，显示更多详情
                treatmentItem.style.cursor = "pointer";
                treatmentItem.addEventListener('click', () => {
                    showDetailedTreatment(treatment);
                });
            }
            
            contentDiv.appendChild(treatmentItem);
            
            // 添加分隔线，最后一项除外
            if (index < treatments.length - 1) {
                const divider = document.createElement('hr');
                divider.style.margin = "10px 0";
                divider.style.border = "none";
                divider.style.borderTop = "1px dashed #e0e0e0";
                contentDiv.appendChild(divider);
            }
        });
        
        recContainer.appendChild(contentDiv);
        
        // 添加查看详情按钮（如果有详情可查看）
        if (treatments.length > 0 && typeof treatments[0] === 'object') {
            const detailsButton = document.createElement('button');
            detailsButton.className = 'action-button secondary';
            detailsButton.style.marginTop = "10px";
            detailsButton.innerHTML = '<i class="fas fa-info-circle"></i> 查看完整治疗方案';
            detailsButton.onclick = function() {
                // 如果有植物类型信息，显示植物病害库
                if (appState.detectedPlantType) {
                    // 找出主要检测到的病害名称
                    let mainDiseaseName = null;
                    if (appState.results && appState.results.detections && appState.results.detections.length > 0) {
                        // 按置信度排序，获取最高置信度的病害
                        const sortedDetections = [...appState.results.detections].sort((a, b) => b.score - a.score);
                        if (sortedDetections[0]) {
                            // 提取病害名称部分（如果是"植物-病害"格式）
                            const fullName = sortedDetections[0].class_name;
                            if (fullName.includes('-')) {
                                mainDiseaseName = fullName.split('-')[1].trim();
                            } else {
                                mainDiseaseName = fullName;
                            }
                        }
                    }
                    
                    // 将检测到的主要病害名称传递给植物病害库函数
                    showPlantDiseaseLibrary(appState.detectedPlantType, mainDiseaseName);
                } else {
                    // 否则显示第一个治疗方案详情
                    showDetailedTreatment(treatments[0]);
                }
            };
            recContainer.appendChild(detailsButton);
        }
        
        container.appendChild(recContainer);
        
        // 添加样式
        const styles = document.createElement('style');
        styles.innerHTML = `
            .high-urgency {
                background-color: #ffebee;
                color: #c62828;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 12px;
            }
            .medium-urgency {
                background-color: #fff8e1;
                color: #ff8f00;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 12px;
            }
            .low-urgency {
                background-color: #e8f5e9;
                color: #2e7d32;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 12px;
            }
            .treatment-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .more-treatments {
                font-size: 12px;
                color: #2196F3;
                margin-top: 5px;
            }
        `;
        document.head.appendChild(styles);
    }

    // 添加一个函数来显示操作提示
    function showTreatmentTip(container) {
        // 创建提示元素
        const tipElement = document.createElement('div');
        tipElement.className = 'treatment-tip';
        tipElement.style.padding = '10px';
        tipElement.style.marginTop = '10px';
        tipElement.style.backgroundColor = '#e3f2fd';
        tipElement.style.borderRadius = '4px';
        tipElement.style.borderLeft = '4px solid #2196F3';
        tipElement.style.fontSize = '14px';
        
        tipElement.innerHTML = `
            <p><i class="fas fa-lightbulb" style="color: #ffc107; margin-right: 5px;"></i> 
            <strong>提示:</strong> 点击上方的"治疗方案"卡片可查看详细治疗信息，或点击可视化区域中的检测框查看对应病害的治疗方案。</p>
            <button id="dismiss-tip" style="background: none; border: none; color: #2196F3; cursor: pointer; padding: 5px; float: right;">
                关闭提示
            </button>
        `;
        
        // 添加到容器
        container.appendChild(tipElement);
        
        // 添加关闭提示的事件监听
        document.getElementById('dismiss-tip').addEventListener('click', function() {
            tipElement.style.display = 'none';
            // 保存状态到localStorage，之后不再显示
            localStorage.setItem('treatment-tip-dismissed', 'true');
        });
    }

    // 修改showPlantDiseaseLibrary函数

    function showPlantDiseaseLibrary(plantType) {
        // 创建对话框
        const modal = document.getElementById('treatment-modal');
        const modalBody = document.getElementById('modal-body');
        
        // 显示加载中
        modalBody.innerHTML = '<div class="loading">加载中...</div>';
        modal.style.display = 'block';
        
        // 获取植物病害信息
        fetch(`/api/plant_diseases?plant=${encodeURIComponent(plantType)}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    modalBody.innerHTML = `<div class="error">${data.error}</div>`;
                    return;
                }
                
                // 显示植物疾病信息
                let htmlContent = `
                    <h3>${plantType} - 病害治疗库</h3>
                    <div class="overview">
                        <p>${data.overview?.description || '暂无概述'}</p>
                    </div>
                `;
                
                // 添加病害列表
                if (data.diseases && data.diseases.length > 0) {
                    htmlContent += `<div class="diseases-list">`;
                    data.diseases.forEach(disease => {
                        htmlContent += `
                            <div class="disease-item">
                                <h4>${disease.name}</h4>
                                <div class="disease-details">
                                    <div class="symptoms">
                                        <h5>症状:</h5>
                                        <ul>
                                            ${(disease.symptoms || []).map(s => `<li>${s}</li>`).join('')}
                                        </ul>
                                    </div>
                                    <div class="treatments">
                                        <h5>治疗方法:</h5>
                                        <ul>
                                            ${(disease.treatments || []).map(t => `<li>${t}</li>`).join('')}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                    htmlContent += `</div>`;
                } else {
                    htmlContent += `<p>没有找到该植物的病害信息</p>`;
                }
                
                modalBody.innerHTML = htmlContent;
            })
            .catch(error => {
                console.error('获取植物病害信息失败:', error);
                modalBody.innerHTML = `<div class="error">获取植物病害信息失败</div>`;
            });
    }

    // 添加模态框关闭按钮事件处理
    document.addEventListener('DOMContentLoaded', function() {
        // 获取所有关闭按钮
        const closeButtons = document.querySelectorAll('.modal .close');
        
        // 添加点击事件
        closeButtons.forEach(button => {
            button.addEventListener('click', function() {
                // 获取父模态框
                const modal = this.closest('.modal');
                if (modal) {
                    modal.style.display = 'none';
                }
            });
        });
        
        // 点击模态框外部关闭
        window.addEventListener('click', function(event) {
            const modals = document.querySelectorAll('.modal');
            modals.forEach(modal => {
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            });
        });
    });

    // 添加函数，用于显示植物病害治疗库
    function showPlantDiseaseLibrary(plantType, detectedDisease = null) {
        // 创建加载指示器
        showLoadingIndicator();
        
        // 请求该植物的所有病害信息
        fetch(`/api/plant_diseases?plant=${encodeURIComponent(plantType)}`)
            .then(response => response.json())
            .then(data => {
                // 隐藏加载指示器
                hideLoadingIndicator();
                
                // 创建疾病库容器
                const resultsContainer = document.getElementById('results-container');
                
                // 清除现有内容
                resultsContainer.innerHTML = '';
                
                // 创建疾病库 DOM 结构
                const libraryContainer = document.createElement('div');
                libraryContainer.className = 'disease-library-container';
                
                // 添加标题和关闭按钮
                libraryContainer.innerHTML = `
                    <div class="disease-library-header">
                        <h3>${plantType}的病害治疗库</h3>
                        <button class="disease-library-close">&times;</button>
                    </div>
                `;
                
                // 如果检测到了特定病害，在顶部显示该病害信息
                if (detectedDisease && data.diseases && data.diseases.length > 0) {
                    const detectedDiseaseInfo = data.diseases.find(d => d.name === detectedDisease);
                    
                    if (detectedDiseaseInfo) {
                        const currentDiseaseSection = document.createElement('div');
                        currentDiseaseSection.className = 'detected-disease-section';
                        
                        currentDiseaseSection.innerHTML = `
                            <h4 class="detected-disease-title">
                                <i class="fas fa-exclamation-circle"></i> 检测到的病害
                            </h4>
                            <div class="detected-disease-card">
                                <h4>${detectedDiseaseInfo.name}</h4>
                                <div class="symptoms-summary">${detectedDiseaseInfo.symptoms_summary}</div>
                                <div class="severity ${getSeverityClass(detectedDiseaseInfo.severity)}">
                                    严重程度: ${detectedDiseaseInfo.severity || '未知'}
                                </div>
                                 <button class="action-button primary view-treatment-btn" 
                                    onclick="fetch('/api/treatment?plant=${encodeURIComponent(plantType)}&disease=${encodeURIComponent(detectedDiseaseInfo.name)}')
                                        .then(res => res.json())
                                        .then(data => showDetailedTreatment(data))
                                        .catch(err => console.error('获取治疗信息失败:', err))">
                                    查看详细治疗方案
                                </button>
                            </div>
                        `;
                        
                        libraryContainer.appendChild(currentDiseaseSection);
                    }
                }
                
                // 如果有概览信息，添加植物概览
                if (data.overview) {
                    const overviewSection = document.createElement('div');
                    overviewSection.className = 'plant-overview';
                    
                    let overviewHTML = `<h4>${plantType}概述</h4>`;
                    
                    if (data.overview.description) {
                        overviewHTML += `<p>${data.overview.description}</p>`;
                    }
                    
                    if (data.overview.general_care && data.overview.general_care.length > 0) {
                        overviewHTML += `
                            <h5>一般护理建议:</h5>
                            <ul>
                                ${data.overview.general_care.map(tip => `<li>${tip}</li>`).join('')}
                            </ul>
                        `;
                    }
                    
                    overviewSection.innerHTML = overviewHTML;
                    libraryContainer.appendChild(overviewSection);
                }
                
                // 添加其他可能的病害列表
                if (data.diseases && data.diseases.length > 0) {
                    const otherDiseasesSection = document.createElement('div');
                    otherDiseasesSection.className = 'other-diseases-section';
                    
                    otherDiseasesSection.innerHTML = `
                        <h4 class="other-diseases-title">其他可能的病害</h4>
                    `;
                    
                    const diseasesDiv = document.createElement('div');
                    diseasesDiv.className = 'disease-items';
                    
                    // 过滤掉已经在顶部显示的检测到的疾病
                    const otherDiseases = detectedDisease ? 
                        data.diseases.filter(d => d.name !== detectedDisease) : 
                        data.diseases;
                    
                    otherDiseases.forEach(disease => {
                        // 为每个疾病创建一个项目
                        const diseaseItem = document.createElement('div');
                        diseaseItem.className = 'disease-item';
                        diseaseItem.dataset.disease = disease.name;
                        
                        // 添加疾病信息
                        let severityClass = getSeverityClass(disease.severity);
                        
                        diseaseItem.innerHTML = `
                            <h4>${disease.name}</h4>
                            <div class="symptoms-summary">${disease.symptoms_summary}</div>
                            <div class="severity ${severityClass}">严重程度: ${disease.severity || '未知'}</div>
                            ${disease.has_treatment ? '<span class="has-treatment">有治疗方案</span>' : ''}
                        `;
                        
                        // 添加点击事件，显示详细治疗信息
                        diseaseItem.addEventListener('click', () => {
                            // 请求特定疾病的详细治疗信息
                            fetch(`/api/treatment?plant=${encodeURIComponent(plantType)}&disease=${encodeURIComponent(disease.name)}`)
                                .then(res => res.json())
                                .then(treatmentData => {
                                    // 显示详细治疗信息
                                    showDetailedTreatment(treatmentData);
                                })
                                .catch(err => {
                                    console.error('获取治疗信息失败:', err);
                                    updateStatusMessage('获取治疗信息失败，请重试', 'error');
                                });
                        });
                        
                        diseasesDiv.appendChild(diseaseItem);
                    });
                    
                    otherDiseasesSection.appendChild(diseasesDiv);
                    libraryContainer.appendChild(otherDiseasesSection);
                } else {
                    // 如果没有疾病信息，显示空状态
                    const emptyState = document.createElement('div');
                    emptyState.className = 'empty-state';
                    emptyState.innerHTML = `
                        <i class="fas fa-leaf"></i>
                        <h4>暂无病害信息</h4>
                        <p>该植物尚未收录病害及治疗信息</p>
                    `;
                    libraryContainer.appendChild(emptyState);
                }
                
                // 添加返回按钮
                const backButtonSection = document.createElement('div');
                backButtonSection.className = 'library-navigation';
                backButtonSection.innerHTML = `
                    <button class="action-button secondary back-to-results">
                        <i class="fas fa-arrow-left"></i> 返回检测结果
                    </button>
                `;
                libraryContainer.appendChild(backButtonSection);
                
                // 添加关闭按钮和返回按钮事件
                resultsContainer.appendChild(libraryContainer);
                
                const closeBtn = libraryContainer.querySelector('.disease-library-close');
                if (closeBtn) {
                    closeBtn.addEventListener('click', () => {
                        resultsContainer.removeChild(libraryContainer);
                        // 如果有保存的检测结果，则重新显示
                        if (appState.results) {
                            displayResults(appState.results);
                        }
                    });
                }
                
                // 返回按钮事件
                const backBtn = libraryContainer.querySelector('.back-to-results');
                if (backBtn) {
                    backBtn.addEventListener('click', () => {
                        resultsContainer.removeChild(libraryContainer);
                        // 重新显示检测结果
                        if (appState.results) {
                            displayResults(appState.results);
                        }
                    });
                }
            })
            .catch(error => {
                hideLoadingIndicator();
                console.error('获取植物病害库失败:', error);
                updateStatusMessage('获取植物病害库失败，请重试', 'error');
            });
    }

    // 添加一个辅助函数用于根据严重程度获取对应的CSS类
    function getSeverityClass(severity) {
        if (!severity) return 'unknown';
        
        if (severity.toLowerCase().includes('严重')) return 'severe';
        if (severity.toLowerCase().includes('中等')) return 'moderate';
        if (severity.toLowerCase().includes('轻微')) return 'mild';
        
        return 'unknown';
    }

    // 页面加载完成后初始化应用
    document.addEventListener('DOMContentLoaded', function() {
        // 初始化应用主要功能
        initializeApp();
        
        // 初始化模态框
        initializeModals();
        
        // 初始化可视化组件事件监听
        setupVisualizationEvents();
        
        // 检查是否需要显示帮助提示
        checkAndShowHelp();
    });

    /**
     * 初始化模态框功能
     */
    function initializeModals() {
        const modal = document.getElementById('treatment-modal');
        const disclaimerModal = document.getElementById('disclaimer-modal');
        
        // 当用户点击模态框的关闭按钮或模态框外部区域时关闭模态框
        const closeButtons = document.querySelectorAll('.modal .close');
        closeButtons.forEach(button => {
            button.addEventListener('click', function() {
                const modal = this.closest('.modal');
                if (modal) {
                    modal.style.display = 'none';
                }
            });
        });
        
        // 点击模态框外部关闭
        window.addEventListener('click', function(event) {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
            if (event.target === disclaimerModal) {
                disclaimerModal.style.display = 'none';
            }
        });
        
        // 显示声明链接
        const disclaimerLink = document.getElementById('show-disclaimer');
        if (disclaimerLink) {
            disclaimerLink.addEventListener('click', function(e) {
                e.preventDefault();
                if (disclaimerModal) {
                    disclaimerModal.style.display = 'block';
                }
            });
        }
    }

    /**
     * 设置可视化事件监听
     */
    function setupVisualizationEvents() {
        // 监听可视化阈值更新
        document.addEventListener('visualization-threshold-update', function(e) {
            if (appState.currentTask === 'detection' && appState.results) {
                // 根据新阈值更新可视化
                updateVisualizationWithThreshold(e.detail.threshold);
                
                // 实时更新检测结果显示
                updateDetectionResultsWithThreshold(e.detail.threshold);
            }
        });
        
        // 监听可视化模式变化
        document.addEventListener('mode-change', function(e) {
            if (appState.currentTask === 'detection' && appState.results) {
                // 切换可视化模式
                updateVisualizationMode(e.detail.mode);
            }
        });
    }

    /**
     * 根据阈值更新检测结果显示
     */
    function updateDetectionResultsWithThreshold(threshold) {
        if (!appState.results || !appState.results.detections) return;
        
        // 过滤检测结果
        const filteredDetections = appState.results.detections.filter(det => det.score >= threshold);
        
        // 更新检测结果统计
        const resultsList = document.querySelector('.disease-list');
        if (resultsList) {
            // 清除旧结果
            resultsList.innerHTML = '';
            
            // 添加新结果
            filteredDetections.forEach(detection => {
                const listItem = document.createElement('li');
                listItem.innerHTML = `
                    <span class="disease-name">${detection.class_name}</span>
                    <span class="confidence-value">${Math.round(detection.score * 100)}%</span>
                `;
                resultsList.appendChild(listItem);
            });
            
            // 如果没有结果，显示提示
            if (filteredDetections.length === 0) {
                resultsList.innerHTML = '<li>当前阈值下未检测到病害</li>';
            }
        }
        
        // 更新严重程度描述
        const severitySummary = document.querySelector('.disease-summary p');
        if (severitySummary) {
            if (filteredDetections.length > 0) {
                // 根据过滤后的检测结果更新严重程度描述
                const maxConfidence = Math.max(...filteredDetections.map(d => d.score));
                let severityText = '轻度';
                
                if (maxConfidence > 0.8 || filteredDetections.length > 3) {
                    severityText = '严重';
                } else if (maxConfidence > 0.6 || filteredDetections.length > 1) {
                    severityText = '中度';
                }
                
                severitySummary.textContent = `检测到${filteredDetections.length}处${severityText}病害区域`;
            } else {
                severitySummary.textContent = '当前阈值下未检测到明显病害';
            }
        }
    }

    /**
     * 检查并显示帮助信息
     */
    function checkAndShowHelp() {
        // 检查是否是首次访问
        const hasVisited = localStorage.getItem('hasVisited');
        
        if (!hasVisited) {
            // 显示功能引导
            showFunctionalGuide();
            
            // 标记已访问
            localStorage.setItem('hasVisited', 'true');
        }
    }

    /**
     * 显示功能引导
     */
    function showFunctionalGuide() {
        // 创建引导提示
        const guideElement = document.createElement('div');
        guideElement.className = 'functional-guide';
        guideElement.innerHTML = `
            <div class="guide-header">
                <h4>欢迎使用植物病害检测系统</h4>
                <button id="close-guide">&times;</button>
            </div>
            <div class="guide-content">
                <p>本系统提供以下功能:</p>
                <ul>
                    <li><i class="fas fa-leaf"></i> 植物种类识别</li>
                    <li><i class="fas fa-search"></i> 植物病害检测</li>
                    <li><i class="fas fa-sliders-h"></i> 可调整的置信度阈值</li>
                    <li><i class="fas fa-chart-pie"></i> 多种可视化模式</li>
                    <li><i class="fas fa-medkit"></i> 智能治疗建议</li>
                </ul>
                <p>开始使用: 上传一张植物图像，然后点击"开始分析"按钮。</p>
            </div>
        `;
        
        // 样式设置
        guideElement.style.position = 'fixed';
        guideElement.style.top = '50%';
        guideElement.style.left = '50%';
        guideElement.style.transform = 'translate(-50%, -50%)';
        guideElement.style.backgroundColor = 'white';
        guideElement.style.padding = '20px';
        guideElement.style.borderRadius = '8px';
        guideElement.style.boxShadow = '0 4px 20px rgba(0,0,0,0.15)';
        guideElement.style.zIndex = '1000';
        guideElement.style.maxWidth = '400px';
        
        document.body.appendChild(guideElement);
        
        // 关闭按钮事件
        document.getElementById('close-guide').addEventListener('click', function() {
            document.body.removeChild(guideElement);
        });
        
        // 样式
        const style = document.createElement('style');
        style.textContent = `
            .guide-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid #e0e0e0;
            }
            .guide-header h4 {
                margin: 0;
                color: #2c3e50;
            }
            .guide-header button {
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #7f8c8d;
            }
            .guide-content ul {
                margin: 15px 0;
                padding-left: 20px;
            }
            .guide-content li {
                margin-bottom: 8px;
            }
            .guide-content i {
                margin-right: 8px;
                color: #3498db;
            }
        `;
        document.head.appendChild(style);
    }

    /**
     * 对需要公开的函数进行局部向全局暴露
     */
    // 暴露需要给HTML元素onclick等事件使用的函数
    window.switchToDetection = switchToDetection;
    window.showPlantInfo = showPlantInfo;
    window.showPlantDiseaseLibrary = showPlantDiseaseLibrary;
    window.showDetailedTreatment = showDetailedTreatment;

    /**
     * 处理 Grad-CAM 解释请求
     */
    function handleExplainRequest(event) {
        if (!appState.uploadedImage) {
            updateStatusMessage('请先上传图像', 'warning');
            return;
        }

        const button = event.currentTarget;  // 使用currentTarget确保我们获取到带有事件监听器的元素
        const resultContainer = button.closest('.gradcam-actions');
        const gradcamImageContainer = resultContainer.querySelector('.gradcam-result-container');
        const gradcamImage = gradcamImageContainer.querySelector('.gradcam-image');
        const loadingIndicator = gradcamImageContainer.querySelector('.gradcam-loading');
        const errorDisplay = gradcamImageContainer.querySelector('.gradcam-error');

        // 隐藏之前的错误/图像，显示加载中
        errorDisplay.style.display = 'none';
        gradcamImage.style.display = 'none';
        loadingIndicator.style.display = 'block';
        gradcamImageContainer.style.display = 'block'; // 显示容器
        button.disabled = true; // 禁用按钮防止重复点击

        // 准备表单数据
        const formData = new FormData();
        formData.append('image', appState.uploadedImage);

        // 调用新的 API 端点
        fetch('/api/explain_detection', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none'; // 隐藏加载中
            button.disabled = false; // 恢复按钮

            if (data.error) {
                console.error('Grad-CAM 生成失败:', data.error);
                errorDisplay.textContent = `无法生成解释: ${data.error}`;
                errorDisplay.style.display = 'block';
            } else if (data.gradcam_image) {
                // 显示 Grad-CAM 图像
                gradcamImage.src = data.gradcam_image; // 设置 Base64 Data URL
                gradcamImage.style.display = 'block';
                // 可以考虑添加标题或说明
                const title = gradcamImageContainer.querySelector('h5');
                if (!title) {
                    const newTitle = document.createElement('h5');
                    newTitle.textContent = `关注区域 (预测: ${data.predicted_class_name})`;
                    gradcamImageContainer.insertBefore(newTitle, gradcamImage);
                } else {
                    title.textContent = `关注区域 (预测: ${data.predicted_class_name})`;
                }
            } else {
                errorDisplay.textContent = '未能获取解释图像。';
                errorDisplay.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('调用 Grad-CAM API 出错:', error);
            loadingIndicator.style.display = 'none';
            errorDisplay.textContent = '请求解释时出错，请检查网络或稍后再试。';
            errorDisplay.style.display = 'block';
            button.disabled = false;
        });
    }

})();