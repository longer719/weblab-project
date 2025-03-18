#src/utils/disease_treatments.py
"""
植物病害治疗建议数据库

提供针对不同植物种类及其病害的详细治疗方案和预防建议。
基于PlantVillage数据集中包含的植物种类和病害。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DiseaseTreatmentDatabase:
    """植物病害治疗建议数据库"""
    
    def __init__(self):
        """初始化数据库"""
        # 使用统一的映射服务
        from src.utils.mapping_service import MappingService
        self.mapping_service = MappingService()
        
        # 构建数据库结构：植物类型 -> 病害类型 -> 治疗信息
        self.treatments = self._initialize_database()
        
    def get_treatment(self, plant_type: str, disease_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取特定植物和病害的治疗建议
        
        Args:
            plant_type: 植物类型名称（如"苹果"、"番茄"）或数据集类名（如"Corn___healthy"）
            disease_type: 病害类型名称（如"黑斑病"），若为None则返回该植物所有病害
            
        Returns:
            包含治疗信息的字典
        """
        # 检查是否为数据集类名格式（包含___）
        if "___" in plant_type:
            plant, disease = self.map_dataset_class_to_db(plant_type)
            return self.get_treatment(plant, disease)
        
        # 标准化输入
        plant = self._normalize_name(plant_type)
        
        # 检查植物是否存在于数据库中
        if plant not in self.treatments:
            logger.warning(f"植物 '{plant_type}' 不在治疗数据库中")
            return {"error": f"未找到'{plant_type}'的治疗信息", "plant_name": plant_type}
        
        # 如果未指定疾病，返回植物概览
        if disease_type is None:
            return {
                "plant_name": plant_type, 
                "overview": self.treatments[plant]["overview"],
                "diseases": [k for k in self.treatments[plant].keys() if k != "overview"]
            }
        
        # 标准化疾病名称
        disease = self._normalize_name(disease_type)
        
        # 检查疾病是否存在
        if disease not in self.treatments[plant]:
            logger.warning(f"病害 '{disease_type}' 不在 '{plant_type}' 的治疗数据库中")
            return {"error": f"未找到'{plant_type}'的'{disease_type}'病害治疗信息", 
                    "plant_name": plant_type, 
                    "disease_name": disease_type}
        
        # 返回疾病治疗信息，添加植物名称
        result = self.treatments[plant][disease].copy()
        result["plant_name"] = plant_type
        result["disease_name"] = disease_type
        return result
    
    def get_supported_plants(self) -> List[str]:
        """获取支持的植物列表"""
        return list(self.treatments.keys())
    
    def get_supported_diseases(self, plant_type: Optional[str] = None) -> List[str]:
        """
        获取支持的病害列表
        
        Args:
            plant_type: 可选，如果提供，则只返回该植物的病害
            
        Returns:
            病害名称列表
        """
        if plant_type is None:
            all_diseases = []
            for plant in self.treatments:
                plant_diseases = [d for d in self.treatments[plant].keys() if d != "overview"]
                all_diseases.extend(plant_diseases)
            return list(set(all_diseases))
        else:
            plant = self._normalize_name(plant_type)
            if plant not in self.treatments:
                logger.warning(f"植物 '{plant_type}' 不在治疗数据库中")
                return []
            return [d for d in self.treatments[plant].keys() if d != "overview"]
    
    def _normalize_name(self, name: str) -> str:
        """规范化名称以便比较"""
        if name is None:
            return ""
        return name.lower().strip()
    
    def map_dataset_class_to_db(self, class_name: str) -> Tuple[str, str]:
        """
        将数据集类名映射到数据库中的植物和病害
        
        Args:
            class_name: 数据集类名，格式为 "植物___状态"，例如 "Corn___healthy"
            
        Returns:
            (植物名，病害名)元组，如 ("玉米", "健康") 或 ("玉米", "锈病")
        """
        # 直接使用映射服务
        return self.mapping_service.map_dataset_class_to_db(class_name)
    
    def _initialize_database(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """初始化治疗数据库"""
        # 创建基础数据结构
        db = {}
        
        # === 苹果 ===
        db["苹果"] = {
            "overview": {
                "description": "苹果树对多种真菌性疾病敏感，需要定期监测和预防性措施。",
                "general_care": ["保持良好通风", "适当修剪以减少枝叶密度", "清除落叶"]
            },
            "黑星病": {
                "symptoms": ["叶片上的橄榄色斑点", "果实上的深色凹陷斑点", "严重时叶片早落"],
                "causes": ["由苹果黑星菌(Venturia inaequalis)真菌引起", "在温暖潮湿的条件下传播"],
                "treatments": [
                    "使用铜基或硫基杀菌剂喷洒",
                    "剪除并销毁受感染的枝条和叶片",
                    "在春季芽前及落叶后进行预防性喷洒"
                ],
                "prevention": [
                    "选择抗病品种",
                    "保持果园清洁，清除落叶和残枝",
                    "确保良好通风和阳光照射",
                    "避免过度浇水，尤其是叶面淋湿"
                ],
                "severity": "中度至严重，不治疗会导致产量和品质下降",
                "organic_solutions": ["使用硫磺或波尔多液等有机杀菌剂", "使用精油混合物如茶树油"]
            },
            "黑腐病": {
                "symptoms": ["果实上形成棕色至黑色的腐烂斑点", "叶片上出现褐色斑点和早期落叶"],
                "causes": ["由博特利斯果腐病菌(Botryosphaeria obtusa)引起", "通过伤口侵入植物组织"],
                "treatments": [
                    "使用适当的杀菌剂",
                    "剪除并焚烧受感染的部位",
                    "在休眠期使用铜基喷剂"
                ],
                "prevention": [
                    "避免机械伤害果树",
                    "控制昆虫害虫减少伤口",
                    "在冬季修剪时涂抹伤口保护剂"
                ],
                "severity": "可能严重影响果实质量和产量",
                "organic_solutions": ["使用石灰硫磺混合物", "应用枯草芽孢杆菌生物制剂"]
            },
            "雪松苹果锈病": {
                "symptoms": ["叶片上出现橙色至黄色斑点", "斑点下方形成角状突起物"],
                "causes": ["由柏科植物与苹果树之间传播的真菌引起", "需要两种宿主植物才能完成生命周期"],
                "treatments": [
                    "使用专门针对锈病的杀菌剂",
                    "去除附近的柏树(可选)",
                    "剪除严重感染的叶片"
                ],
                "prevention": [
                    "种植抗锈病品种",
                    "避免在附近种植柏树",
                    "早春预防性喷药"
                ],
                "severity": "通常中等严重性，但可导致早期落叶和树势减弱",
                "organic_solutions": ["使用硫磺喷剂", "大蒜提取物喷雾"]
            }
        }
        
        # === 玉米 ===
        db["玉米"] = {
            "overview": {
                "description": "玉米作为重要粮食作物，易受多种真菌性病害影响，特别是在温暖潮湿的条件下。",
                "general_care": ["轮作", "选择抗病品种", "适当间距以确保通风"]
            },
            "常见锈病": {
                "symptoms": ["叶片两面出现突起的橙红色至棕色斑点", "严重时叶片变黄并干枯"],
                "causes": ["由玉米锈菌(Puccinia sorghi)引起", "在温暖潮湿条件下迅速传播"],
                "treatments": [
                    "喷施推荐的杀菌剂",
                    "去除受感染的植株部分"
                ],
                "prevention": [
                    "种植抗病品种",
                    "轮作3-4年",
                    "避免过度灌溉"
                ],
                "severity": "在适宜条件下可导致20-40%的产量损失",
                "organic_solutions": ["喷施硫磺制剂", "应用植物提取物如芸香科植物提取物"]
            },
            "北方叶枯病": {
                "symptoms": ["长椭圆形灰绿色至棕色病斑", "湿度高时病斑上出现黑色霉层"],
                "causes": ["由玉米大斑病菌(Exserohilum turcicum)引起", "在凉爽多雨季节严重"],
                "treatments": [
                    "及时喷洒杀菌剂",
                    "清除受感染的作物残株"
                ],
                "prevention": [
                    "实行轮作",
                    "种植抗病品种",
                    "使用平衡肥料"
                ],
                "severity": "可能导致50%以上产量损失，特别是早期感染时",
                "organic_solutions": ["使用铜基有机杀菌剂", "应用芽孢杆菌生物防治剂"]
            },
            "叶斑病": {
                "symptoms": ["小的圆形至椭圆形褐色斑点，有黄色晕圈", "多个病斑融合可导致大面积叶片死亡"],
                "causes": ["由玉米灰斑病菌(Cercospora zeae-maydis)引起", "高湿度和连作加重病情"],
                "treatments": [
                    "应用适当杀菌剂",
                    "清除田间残株"
                ],
                "prevention": [
                    "实行玉米与非寄主作物轮作",
                    "深翻土壤覆盖残株",
                    "选择抗性品种"
                ],
                "severity": "严重影响光合作用效率，可导致20-60%产量损失",
                "organic_solutions": ["使用有机铜制剂", "应用微生物拮抗剂"]
            }
        }
        
        # === 番茄 ===
        db["番茄"] = {
            "overview": {
                "description": "番茄是受多种病害影响的敏感作物，尤其在高温高湿条件下。",
                "general_care": ["搭架支撑以改善通风", "早晨浇水避免叶片长时间湿润", "使用地膜减少土传病害"]
            },
            "早疫病": {
                "symptoms": ["叶片上出现暗色同心圆斑点", "茎部和果实可出现深色病斑"],
                "causes": ["由番茄早疫病菌(Alternaria solani)引起", "高温(24-29°C)和高湿条件有利于发病"],
                "treatments": [
                    "使用杀菌剂喷洒，如含铜制剂",
                    "去除并销毁受感染的叶片和植株"
                ],
                "prevention": [
                    "轮作3-4年",
                    "使用抗病品种",
                    "保持适当的植株间距"
                ],
                "severity": "可导致30-50%的产量损失",
                "organic_solutions": ["使用波尔多液", "喷洒大蒜提取物或牛奶稀释液"]
            },
            "晚疫病": {
                "symptoms": ["叶片上出现不规则水渍状病斑，潮湿时有白霉", "果实上出现坚硬的褐色斑点"],
                "causes": ["由致病疫霉(Phytophthora infestans)引起", "凉爽多雨天气特别有利于发病"],
                "treatments": [
                    "使用专门针对卵菌的杀菌剂",
                    "在发现症状后立即处理",
                    "移除受感染的植株以防止传播"
                ],
                "prevention": [
                    "改善通风",
                    "避免过密种植",
                    "使用滴灌而非喷洒灌溉"
                ],
                "severity": "在适宜条件下可导致全部植株死亡，是历史上著名的爱尔兰马铃薯饥荒病原",
                "organic_solutions": ["使用铜制剂", "应用牛至油等精油提取物"]
            },
            "叶霉病": {
                "symptoms": ["叶片上表面出现黄色斑块，背面有霉层", "叶片逐渐枯萎"],
                "causes": ["由叶霉菌(Fulvia fulva)引起", "高湿度(85%以上)特别有利于发病"],
                "treatments": [
                    "使用合适的杀菌剂",
                    "改善温室通风",
                    "控制湿度"
                ],
                "prevention": [
                    "选择抗病品种",
                    "避免叶片长时间潮湿",
                    "适当间距种植"
                ],
                "severity": "主要影响产量，严重时可减产10-50%",
                "organic_solutions": ["使用小苏打溶液喷洒", "应用乳清溶液"]
            },
            "细菌斑点病": {
                "symptoms": ["叶片上出现小黑点，周围有黄晕", "果实上出现小而凹陷的病斑"],
                "causes": ["由番茄细菌性斑点病菌(Xanthomonas campestris)引起", "通过种子、灌溉水和工具传播"],
                "treatments": [
                    "使用铜制杀菌剂",
                    "避免在湿叶上工作",
                    "去除严重感染的植株"
                ],
                "prevention": [
                    "使用健康种子和幼苗",
                    "避免使用喷灌",
                    "实行作物轮作"
                ],
                "severity": "可导致20-30%的产量损失",
                "organic_solutions": ["使用铜制有机杀菌剂", "应用大蒜或辣椒提取物"]
            }
        }
        
        # === 葡萄 ===
        db["葡萄"] = {
            "overview": {
                "description": "葡萄容易受多种真菌病害影响，尤其在温暖潮湿的气候条件下。",
                "general_care": ["定期修剪以改善通风", "合理控制水分", "适当施肥提高抗性"]
            },
            "黑腐病": {
                "symptoms": ["叶片上出现褐色圆形病斑，边缘较深", "果实上出现凹陷病斑，逐渐扩大并变黑"],
                "causes": ["由葡萄黑腐病菌(Guignardia bidwellii)引起", "在温暖潮湿的环境中迅速传播"],
                "treatments": [
                    "定期喷洒杀菌剂",
                    "移除并销毁受感染的部分",
                    "改善通风条件"
                ],
                "prevention": [
                    "种植抗病品种",
                    "移除野生葡萄藤",
                    "春季早期开始预防性喷洒"
                ],
                "severity": "可导致整个葡萄园的葡萄损失",
                "organic_solutions": ["使用硫磺喷剂", "喷施铜制有机杀菌剂"]
            },
            "霜霉病": {
                "symptoms": ["叶片上出现黄色油状斑点，下表面有白色霉层", "幼果受感染后变褐、干缩"],
                "causes": ["由葡萄霜霉菌(Plasmopara viticola)引起", "在温暖潮湿条件下严重发生"],
                "treatments": [
                    "使用专门针对霜霉病的杀菌剂",
                    "及时去除受感染的组织",
                    "改善葡萄园排水和通风"
                ],
                "prevention": [
                    "避免过密种植",
                    "定期修剪改善通风",
                    "在雨季前进行预防性喷洒"
                ],
                "severity": "在潮湿季节可导致严重减产，甚至达80%",
                "organic_solutions": ["使用波尔多液", "喷施海藻提取物增强植株抵抗力"]
            },
            "白粉病": {
                "symptoms": ["叶片和果实表面出现白色粉状覆盖物", "叶片可能变形和早期脱落"],
                "causes": ["由葡萄白粉病菌(Uncinula necator)引起", "在温暖干燥的条件下蔓延"],
                "treatments": [
                    "使用专门针对白粉病的杀菌剂",
                    "硫磺喷剂通常有效",
                    "去除严重感染的枝条"
                ],
                "prevention": [
                    "选择抗病品种",
                    "避免过量使用氮肥",
                    "保持适当的修剪和通风"
                ],
                "severity": "影响光合作用和果实品质，可减产10-65%",
                "organic_solutions": ["使用硫磺粉", "喷洒小苏打溶液", "应用牛奶溶液"]
            },
            "褐斑病": {
                "symptoms": ["叶片上出现不规则红褐色斑点", "严重时叶片干枯脱落"],
                "causes": ["由多种真菌病原体引起", "通常在生长季后期严重"],
                "treatments": [
                    "使用广谱杀菌剂",
                    "去除落叶减少病菌积累",
                    "适当修剪增加通风"
                ],
                "prevention": [
                    "避免叶片长时间潮湿",
                    "合理密植",
                    "平衡施肥"
                ],
                "severity": "主要影响叶片，间接降低产量和品质",
                "organic_solutions": ["使用铜制剂", "喷施茶叶提取物"]
            }
        }

        # === 橙子 ===
        db["橙子"] = {
            "overview": {
                "description": "橙子是柑橘类水果，易受柑橘黄龙病等病害侵袭，需要特别关注。",
                "general_care": ["控制灌溉", "适当施肥", "定期修剪增加通风"]
            },
            "黄龙病": {
                "symptoms": ["叶片黄化，呈不规则斑驳状", "果实变小、变形、偏酸、不均匀着色", "树势衰弱，提前落叶落果"],
                "causes": ["由细菌'利伯杆菌'引起", "通过木虱等昆虫传播"],
                "treatments": [
                    "立即移除并销毁受感染的树",
                    "进行系统性杀虫剂处理控制传播媒介",
                    "加强树势管理提高抵抗力"
                ],
                "prevention": [
                    "使用无病害的种苗",
                    "定期检查树木健康状况",
                    "控制媒介昆虫种群"
                ],
                "severity": "极其严重，目前无有效治愈方法，常导致树木死亡",
                "organic_solutions": ["使用有机杀虫剂如印楝素", "放养天敌控制媒介昆虫"]
            }
        }

        # === 蓝莓 ===
        db["蓝莓"] = {
            "overview": {
                "description": "蓝莓是一种营养价值高的浆果，对土壤pH值要求严格，常见真菌性病害。",
                "general_care": ["保持酸性土壤", "充分灌溉但避免积水", "适量有机质覆盖"]
            },
            "健康": {
                "symptoms": ["叶片翠绿健康", "植株生长旺盛", "浆果饱满均匀着色"],
                "causes": ["适宜的生长环境", "充分的养分供应", "良好的管理"],
                "treatments": [],
                "prevention": [
                    "维持适宜的土壤pH值(4.0-5.5)",
                    "保证充足但不过量的水分",
                    "适当修剪以促进通风"
                ],
                "severity": "健康状态，不需要治疗",
                "organic_solutions": ["使用有机肥料如松针腐殖质", "添加硫磺降低土壤pH值"]
            }
        }

        # === 树莓 ===
        db["树莓"] = {
            "overview": {
                "description": "树莓是一种多年生灌木，果实营养丰富，但易受多种真菌和病毒侵袭。",
                "general_care": ["适当修剪", "搭架支撑", "合理密植"]
            },
            "健康": {
                "symptoms": ["叶片颜色正常有光泽", "茎秆坚挺无病斑", "果实丰满有光泽"],
                "causes": ["合理的栽培管理", "适宜的环境条件", "无病虫害侵袭"],
                "treatments": [],
                "prevention": [
                    "轮作",
                    "使用健康无病的种苗",
                    "保持田间卫生"
                ],
                "severity": "健康状态，无需治疗",
                "organic_solutions": ["使用堆肥增加土壤有机质", "种植伴生植物如大蒜提高抗病性"]
            }
        }

        # === 大豆 ===
        db["大豆"] = {
            "overview": {
                "description": "大豆是重要的油料和蛋白质来源作物，对温度和光照要求较高。",
                "general_care": ["适时播种", "中耕除草", "合理密植"]
            },
            "健康": {
                "symptoms": ["叶片深绿有光泽", "植株生长整齐一致", "荚果饱满"],
                "causes": ["适宜的栽培条件", "良好的土壤肥力", "无有害生物侵袭"],
                "treatments": [],
                "prevention": [
                    "轮作",
                    "适时播种",
                    "使用抗病品种"
                ],
                "severity": "健康状态，无需治疗",
                "organic_solutions": ["接种根瘤菌提高固氮能力", "使用有机肥料"]
            }
        }

        # === 甜椒 ===
        db["甜椒"] = {
            "overview": {
                "description": "甜椒喜温暖环境，对光照和温度要求较高，易受多种病菌侵袭。",
                "general_care": ["适宜温度为20-30℃", "保持土壤湿润但不积水", "充足光照"]
            },
            "细菌斑点病": {
                "symptoms": ["叶片、果实和茎上出现水渍状斑点", "斑点逐渐扩大变为黑褐色", "严重时叶片干枯脱落"],
                "causes": ["由细菌Xanthomonas campestris引起", "通过雨水飞溅、灌溉水和农具传播"],
                "treatments": [
                    "去除并销毁受感染的植物部分",
                    "使用铜制杀菌剂",
                    "避免在潮湿条件下工作"
                ],
                "prevention": [
                    "使用抗病品种",
                    "轮作3-4年",
                    "使用滴灌减少叶片湿度"
                ],
                "severity": "中度到严重，可导致15-30%的产量损失",
                "organic_solutions": ["使用铜肥皂混合液", "应用植物提取物如蒜素喷剂"]
            },
            "健康": {
                "symptoms": ["叶片翠绿无斑点", "茎秆健壮", "果实形状均匀色泽鲜亮"],
                "causes": ["良好的栽培条件", "适当的养分供应", "有效的病虫害管理"],
                "treatments": [],
                "prevention": [
                    "维持均衡施肥",
                    "适度浇水避免叶片长时间潮湿",
                    "定期检查植株健康状况"
                ],
                "severity": "健康状态，无需治疗",
                "organic_solutions": ["使用腐熟堆肥提高土壤肥力", "种植驱虫植物如万寿菊"]
            }
        }

        # === 西葫芦 ===
        db["西葫芦"] = {
            "overview": {
                "description": "西葫芦属于葫芦科，生长迅速，喜温暖环境，但对霜冻敏感。",
                "general_care": ["充足阳光", "定期浇水", "施用有机肥料"]
            },
            "白粉病": {
                "symptoms": ["叶片、茎和果实表面出现白色粉状霉层", "叶片发黄干枯", "植株生长迟缓"],
                "causes": ["由真菌引起", "高湿度和温暖环境促进发病", "干燥条件下风媒传播孢子"],
                "treatments": [
                    "去除并销毁受感染的植物部分",
                    "应用杀菌剂，如硫磺或钾肥皂",
                    "提高通风，减少植株密度"
                ],
                "prevention": [
                    "选择抗病品种",
                    "避免在叶片上浇水",
                    "合理间距种植"
                ],
                "severity": "中等，如不处理可能导致产量下降20-30%",
                "organic_solutions": ["喷洒小苏打溶液(1茶匙/加仑水)", "使用牛奶溶液(1:10稀释)", "喷洒大蒜提取物"]
            }
        }

        # 添加植物描述和特点的完整性说明
        for plant in ["苹果", "玉米", "葡萄", "番茄"]:
            if plant in db and "overview" in db[plant]:
                db[plant]["overview"]["detailed_info"] = {
                    "生长条件": "参见农业手册了解详细种植条件",
                    "常见品种": "请查阅当地农业部门推荐品种",
                    "经济价值": "作为重要的经济作物，具有较高的市场价值"
                }

        # 完善草莓的信息
        db["草莓"] = {
            "overview": {
                "description": "草莓是多年生草本植物，喜温暖气候，对土壤要求不严，但需良好排水。",
                "general_care": ["定期浇水但避免叶片湿润", "覆盖地面防杂草和保持果实清洁", "适当施肥尤其是开花前"]
            },
            "叶焦病": {
                "symptoms": ["叶缘和叶尖开始变褐色", "逐渐向叶中央发展形成V形病斑", "严重时整片叶子干枯"],
                "causes": ["由真菌Diplocarpon earlianum引起", "通过雨水飞溅和风传播", "高温高湿条件下传播迅速"],
                "treatments": [
                    "移除并销毁受感染的叶片",
                    "使用真菌杀菌剂",
                    "改善田间通风"
                ],
                "prevention": [
                    "选择抗病品种",
                    "避免高密度种植",
                    "采用滴灌而非喷灌"
                ],
                "severity": "中度到严重，可影响产量和果实质量",
                "organic_solutions": ["喷洒石灰硫磺混合物", "使用堆肥茶灌溉增强植株抵抗力"]
            },
            "健康": {
                "symptoms": ["叶片浓绿有光泽", "茎叶丰满健壮", "花朵和果实发育正常"],
                "causes": ["良好的栽培管理", "适宜的生长环境", "有效的病虫害防控"],
                "treatments": [],
                "prevention": [
                    "定期更换种植地点(3-4年一次)",
                    "使用健康的种苗",
                    "秋季清除和销毁老叶"
                ],
                "severity": "健康状态，无需治疗",
                "organic_solutions": ["使用秸秆覆盖保持果实清洁", "应用海藻肥增强植株活力"]
            }
        }

        #继续添加plantvillage数据集中的植物和病害
        
        return db

    def get_plant_info(self, plant_type: str) -> Dict[str, Any]:
        """
        获取植物的详细信息（不包括病害信息）
        
        Args:
            plant_type: 植物类型名称
            
        Returns:
            植物信息字典
        """
        # 标准化植物名称
        plant = self._normalize_name(plant_type)
        
        # 检查植物是否存在
        if plant not in self.treatments:
            return {
                "description": f"未找到关于{plant_type}的详细信息",
                "general_care": []
            }
        
        # 返回植物概览信息
        result = {}
        if "overview" in self.treatments[plant]:
            result = self.treatments[plant]["overview"].copy()
        
        # 添加常见病害列表
        result["common_diseases"] = []
        for disease in self.treatments[plant]:
            if disease != "overview" and isinstance(self.treatments[plant][disease], dict):
                result["common_diseases"].append(disease)
        
        return result

    def is_disease_compatible_with_plant(self, plant_type: str, disease_name: str) -> bool:
        """检查病害是否与特定植物兼容"""
        # "健康"状态与所有植物兼容
        if disease_name == "健康":
            return True
            
        # 植物-病害兼容性表（根据实际情况调整）
        compatibility = {
            "苹果": ["黑星病", "黑腐病", "雪松苹果锈病"],
            "樱桃": ["白粉病"],
            "玉米": ["灰斑病", "普通锈病", "北方叶枯病"],
            "葡萄": ["黑腐病", "黑麻疹病", "叶枯病"],
            "橙子": ["黄龙病"],
            "桃子": ["细菌性斑点病"],
            "甜椒": ["细菌性斑点病"],
            "土豆": ["早疫病", "晚疫病"],
            "草莓": ["叶焦病"],
            "番茄": ["细菌性斑点病", "早疫病", "晚疫病", "叶霉病", "斑枯病", "二斑叶螨", "靶斑病", "花叶病毒病", "黄化曲叶病毒病"]
        }
        
        # 标准化名称以便比较
        norm_plant = plant_type.lower().strip()
        norm_disease = disease_name.lower().strip()
        
        # 检查兼容性
        for plant, diseases in compatibility.items():
            if plant.lower() in norm_plant or norm_plant in plant.lower():
                return any(d.lower() in norm_disease or norm_disease in d.lower() for d in diseases)
        
        return False  # 如果找不到匹配，默认不兼容