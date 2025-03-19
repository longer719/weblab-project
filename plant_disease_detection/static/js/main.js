//这是一个前端脚本文件，用于处理用户上传的图像并将其发送到后端进行处理。
//它还负责显示处理结果和处理用户与结果相关的操作。
/**
 * 植物种类识别与病害检测系统前端交互脚本
 */

// 全局配置和状态
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

// 应用状态
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
    
    // 使用纯植物名称(已经在后端提取)
    const plantType = topPrediction.class_name;
    
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
                    ${results.predictions.slice(1, 4).map(pred => `
                        <li>
                            <span class="prediction-name">${pred.class_name}</span>
                            <span class="prediction-confidence">${(pred.confidence * 100).toFixed(2)}%</span>
                        </li>
                    `).join('')}
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
    // 创建检测结果卡片 (保留原有卡片显示)
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card detection-result';
    
    // 获取检测结果数据
    const detections = results.detections || [];
    const severityInfo = results.severity_assessment || { level: '未知', description: '无法评估' };
    
    // 显示检测结果和严重程度
    resultCard.innerHTML = `
        <div class="result-header">
            <h4>检测到 ${detections.length} 个病害区域</h4>
            <div class="severity-badge ${severityInfo.level ? severityInfo.level.toLowerCase() : 'unknown'}">
                严重程度: ${severityInfo.level || '未知'}
            </div>
        </div>
        <div class="result-details">
            <div class="detection-summary">
                <h5>病害摘要：</h5>
                <p>${severityInfo.description || '无详细描述'}</p>
                <ul class="disease-list">
                    ${detections.map(det => `
                        <li>
                            <span class="disease-name">${det.class_name}</span>
                            <span class="disease-confidence">${(det.score * 100).toFixed(2)}%</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        </div>
    `;
    
    // 添加到容器
    container.appendChild(resultCard);
    
    // 添加治疗信息卡片（如果有）
    if (results.treatment_info) {
        const treatmentCard = document.createElement('div');
        treatmentCard.className = 'treatment-card';
        
        // 获取治疗信息
        const treatment = results.treatment_info;
        const plantName = treatment.plant_name || '未知植物';
        const diseaseName = treatment.disease_name || '未知病害';
        
        // 构建治疗信息HTML
        let treatmentHtml = `
            <div class="treatment-header">
                <h4>${plantName}的${diseaseName}治疗方案</h4>
            </div>
            <div class="treatment-content">`;
        
        // 添加症状部分
        if (treatment.symptoms && treatment.symptoms.length > 0) {
            treatmentHtml += `
                <section>
                    <h5>症状表现：</h5>
                    <ul class="symptom-list">
                        ${treatment.symptoms.map(symptom => `<li>${symptom}</li>`).join('')}
                    </ul>
                </section>`;
        }
        
        // 添加病因部分
        if (treatment.causes && treatment.causes.length > 0) {
            treatmentHtml += `
                <section>
                    <h5>病害原因：</h5>
                    <p>${treatment.causes.join(', ')}</p>
                </section>`;
        }
        
        // 添加治疗方法部分
        if (treatment.treatments && treatment.treatments.length > 0) {
            treatmentHtml += `
                <section>
                    <h5>推荐治疗方法：</h5>
                    <ul class="treatment-list">
                        ${treatment.treatments.map(method => `<li>${method}</li>`).join('')}
                    </ul>
                </section>`;
        }
        
        // 添加预防措施部分
        if (treatment.prevention && treatment.prevention.length > 0) {
            treatmentHtml += `
                <section>
                    <h5>预防措施：</h5>
                    <ul class="prevention-list">
                        ${treatment.prevention.map(tip => `<li>${tip}</li>`).join('')}
                    </ul>
                </section>`;
        }
        
        treatmentHtml += `</div>`;
        
        // 设置卡片内容
        treatmentCard.innerHTML = treatmentHtml;
        
        // 添加点击事件，显示详细信息
        treatmentCard.addEventListener('click', function() {
            showDetailedTreatment(treatment);
        });
        
        // 添加到容器
        container.appendChild(treatmentCard);
        
        // 显示治疗提示（如果用户之前没有关闭过）
        if (!localStorage.getItem('treatment-tip-dismissed')) {
            showTreatmentTip(container);
        }
    } else if (results.treatment_recommendations && results.treatment_recommendations.length > 0) {
        // 如果有治疗建议但没有完整治疗信息，显示简化版本
        const recommendationsDiv = document.createElement('div');
        recommendationsDiv.className = 'treatment-recommendations';
        recommendationsDiv.innerHTML = `
            <h5>治疗建议：</h5>
            <ul>
                ${results.treatment_recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        `;
        container.appendChild(recommendationsDiv);
    }
    
    // 添加可视化容器
    const visContainer = document.createElement('div');
    visContainer.id = 'visualization-container';
    visContainer.className = 'visualization-container';
    container.appendChild(visContainer);
    
    // 准备用于可视化的图像对象
    const originalImage = document.querySelector('.preview-image');
    
    if (originalImage && detections.length > 0) {
        // 如果已经加载了PlantVis可视化组件，则使用它显示检测框
        if (typeof PlantVis !== 'undefined') {
            // 准备检测框数据
            const boxData = detections.map(det => ({
                box: [det.bbox.x, det.bbox.y, det.bbox.x + det.bbox.width, det.bbox.y + det.bbox.height],
                score: det.score,
                class_name: det.class_name,
                severity: det.severity || 'unknown'
            }));
            
            // 渲染检测框
            const canvas = PlantVis.renderDetectionBoxes(originalImage, boxData);
            PlantVis.displayVisualization(visContainer, canvas, {
                interactive: true,
                allowThresholdChange: true,
                visModes: ['boxes', 'heatmap']
            });
        }
    }
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
    
    const detections = appState.results.detections;
    const formattedDetections = detections.map(det => ({
        box: [det.bbox.x, det.bbox.y, det.bbox.x + det.bbox.width, det.bbox.y + det.bbox.height],
        score: det.score,
        class_name: det.class_name
    }));
    
    // 使用PlantVis更新可视化，应用新阈值
    const visContainer = document.getElementById('visualization-container');
    if (visContainer && typeof PlantVis !== 'undefined') {
        const canvas = PlantVis.renderDetectionBoxes(
            originalImage,
            formattedDetections,
            { 
                showLabels: true, 
                showScores: true,
                scoreThreshold: threshold 
            }
        );
        
        PlantVis.displayVisualization(visContainer, canvas, {
            allowThresholdChange: true,
            initialThreshold: threshold,
            visModes: [
                { label: '边界框', value: 'boxes' },
                { label: '热图', value: 'heatmap' }
            ],
            currentMode: 'boxes'
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
    
    // 根据模式选择不同的渲染方法
    if (mode === 'boxes') {
        updateVisualizationWithThreshold(0.5); // 使用默认阈值重新渲染边界框
    } else if (mode === 'heatmap') {
        // 生成热图数据
        const detections = appState.results.detections;
        const heatmapData = generateHeatmapData(detections, originalImage.width, originalImage.height);
        
        // 渲染热图
        const canvas = PlantVis.renderHeatmap(originalImage, heatmapData);
        PlantVis.displayVisualization(visContainer, canvas, {
            visModes: [
                { label: '边界框', value: 'boxes' },
                { label: '热图', value: 'heatmap' }
            ],
            currentMode: 'heatmap'
        });
    }
}

/**
 * 根据检测结果生成热图数据
 */
function generateHeatmapData(detections, width, height) {
    // 简化版热图数据生成
    return detections.map(det => {
        const centerX = det.bbox.x + det.bbox.width / 2;
        const centerY = det.bbox.y + det.bbox.height / 2;
        const radius = Math.max(det.bbox.width, det.bbox.height) / 2;
        
        return {
            x: centerX,
            y: centerY,
            value: det.score, // 使用置信度作为热点强度
            radius: radius * 1.5 // 稍微扩大半径使热图更明显
        };
    });
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
    modalBody.innerHTML = `
        <h3>${treatment.plant_name}的${treatment.disease_name}详细治疗方案</h3>
        
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
    `;
    
    // 显示模态框
    modal.style.display = 'block';
}

// 添加处理治疗建议的辅助函数
function processTreatmentRecommendations(treatments, container) {
    if (!treatments || treatments.length === 0) {
        return;
    }
    
    const recContainer = document.createElement('div');
    recContainer.className = 'treatment-recommendations';
    
    let content = '<h5>推荐治疗方法：</h5><ul>';
    
    // 根据建议类型处理不同的显示方式
    treatments.forEach(treatment => {
        if (typeof treatment === 'string') {
            // 简单字符串建议
            content += `<li>${treatment}</li>`;
        } else if (typeof treatment === 'object') {
            // 结构化建议对象
            content += `<li><strong>${treatment.title || '治疗方法'}:</strong> ${treatment.description || treatment.method || '无详细说明'}</li>`;
        }
    });
    
    content += '</ul>';
    recContainer.innerHTML = content;
    
    // 添加查看详情按钮（如果有详情可查看）
    if (container.querySelector('.treatment-card')) {
        const detailsButton = document.createElement('button');
        detailsButton.className = 'action-button secondary';
        detailsButton.innerHTML = '<i class="fas fa-info-circle"></i> 查看详细治疗方案';
        detailsButton.onclick = function() {
            const treatmentCard = container.querySelector('.treatment-card');
            if (treatmentCard) {
                treatmentCard.scrollIntoView({ behavior: 'smooth' });
                // 添加高亮效果
                treatmentCard.classList.add('highlight');
                setTimeout(() => {
                    treatmentCard.classList.remove('highlight');
                }, 1500);
            }
        };
        recContainer.appendChild(detailsButton);
    }
    
    container.appendChild(recContainer);
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
function showPlantDiseaseLibrary(plantType) {
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
            
            // 添加疾病列表
            if (data.diseases && data.diseases.length > 0) {
                const diseasesDiv = document.createElement('div');
                diseasesDiv.className = 'disease-items';
                
                data.diseases.forEach(disease => {
                    // 为每个疾病创建一个项目
                    const diseaseItem = document.createElement('div');
                    diseaseItem.className = 'disease-item';
                    diseaseItem.dataset.disease = disease.name;
                    
                    // 添加疾病信息
                    let severityClass = 'unknown';
                    if (disease.severity) {
                        if (disease.severity.toLowerCase().includes('严重')) severityClass = 'severe';
                        else if (disease.severity.toLowerCase().includes('中等')) severityClass = 'moderate';
                        else if (disease.severity.toLowerCase().includes('轻微')) severityClass = 'mild';
                    }
                    
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
                
                libraryContainer.appendChild(diseasesDiv);
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
            
            // 添加关闭按钮事件
            resultsContainer.appendChild(libraryContainer);
            
            const closeBtn = libraryContainer.querySelector('.disease-library-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', () => {
                    resultsContainer.removeChild(libraryContainer);
                });
            }
        })
        .catch(error => {
            hideLoadingIndicator();
            console.error('获取植物病害信息失败:', error);
            updateStatusMessage('获取植物病害信息失败，请重试', 'error');
        });
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    
    // 初始化弹窗功能
    const modal = document.getElementById('treatment-modal');
    const disclaimerModal = document.getElementById('disclaimer-modal');
    
    // 当用户点击模态框的关闭按钮或模态框外部区域时关闭模态框
    document.querySelectorAll('.modal .close').forEach(closeBtn => {
        closeBtn.addEventListener('click', function() {
            this.closest('.modal').style.display = 'none';
        });
    });
    
    // 点击模态框外部区域关闭
    window.addEventListener('click', function(event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';
        }
    });
    
    // 打开免责声明弹窗
    document.getElementById('show-disclaimer').addEventListener('click', function(e) {
        e.preventDefault();
        disclaimerModal.style.display = 'block';
    });
});