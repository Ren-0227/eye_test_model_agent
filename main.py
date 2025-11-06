# main.py
from flask import Flask, request, jsonify
import sys
from multiprocessing import Process, Queue
import requests
import json
import os
import datetime
import cv2
from api_integration import DeepseekAPI  # 使用新的API调用方式
from image_processing import analyze_image  # 使用新的图片识别模块
from memory_manager import get_user_memory, update_user_memory
from PyQt5.QtWidgets import QApplication
from vision_test import VisionTester  # 导入VisionTester类

# 常量定义
REPORT_DIR = "replay"
os.makedirs(REPORT_DIR, exist_ok=True)

# 初始化Flask应用
app = Flask(__name__)

def process_image(image_path):
    """处理图片并返回分析结果"""
    try:
        # 确保图片路径有效
        if not os.path.exists(image_path):
            return "图片分析失败: 图片文件未找到，请检查路径是否正确"
        
        # 调用图片分析函数并返回中文标签
        result = analyze_image(image_path)
        return f"检测结果: {result}"
    except Exception as e:
        return f"图片分析失败: {str(e)}"

class EyeHealthSystem:
    def __init__(self):
        self.api = DeepseekAPI()  # 使用新的API集成
        self.vision_tester = VisionTester()  # 初始化VisionTester类
        self.vision_result = None
        self.conversation_history = []
        self.awaiting_test = False

    def needs_vision_test(self, symptoms):
        """优化后的视力检测判断逻辑"""
        vision_keywords = [
            '模糊', '近视', '远视', '看不清', 
            '视力下降', '眼睛疲劳', '眯眼'
        ]
        return any(kw in symptoms for kw in vision_keywords)

    def run_vision_test(self):
        """运行视力检测并返回结果"""
        try:
            # 调用VisionTester的run_test方法
            result = self.vision_tester.run_test()
            return result
        except Exception as e:
            print(f"测试异常：{str(e)}")
            return None

    def handle_image_input(self, image_path):
        """处理图片输入并显示分析区域"""
        try:
            # 确保图片路径有效
            if not os.path.exists(image_path):
                return "图片文件未找到，请检查路径是否正确"
            
            img = cv2.imread(image_path)
            if img is None:
                return "无法读取图片，请检查路径是否正确"
            
            # 显示图片并添加分析标记
            marked_img = self._mark_analysis_areas(img.copy())
            cv2.imshow('眼部分析', marked_img)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            
            return "已接收眼部图片，正在分析..."
        except Exception as e:
            return f"图片处理出错: {str(e)}"

    def _mark_analysis_areas(self, img):
        """在图片上标记分析区域"""
        h, w = img.shape[:2]
        cv2.rectangle(img, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        cv2.putText(img, "分析区域", (w//4, h//4-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img

    def start(self):
        """启动命令行交互界面"""
        print("=== 眼科健康辅助系统 ===")
        print("功能说明:")
        print("1. 文字咨询 - 输入症状描述获取建议")
        print("2. 图片分析 - 上传眼部照片进行分析")
        
        while True:
            try:
                input_type = input("请选择输入方式 (1文字/2图片/q退出): ").lower()
                
                if input_type == 'q':
                    print("感谢使用，再见！")
                    break
        
                if input_type == '1':
                    self._handle_text_input()
                elif input_type == '2':
                    self._handle_image_input()
                else:
                    print("无效输入，请重新选择")
            except KeyboardInterrupt:
                print("\n检测到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"发生错误: {str(e)}")
                continue

    def _handle_text_input(self):
        """处理多轮文本输入逻辑"""
        print("\n进入咨询模式（输入'退出'结束咨询）")
        self.conversation_history = []
        
        while True:
            user_input = input("\n患者: ").strip()
            if user_input.lower() in ['退出', 'q']:
                break
            
            # 将用户输入加入对话历史
            self.conversation_history.append({"role": "user", "content": user_input})
            
            try:
                # 调用API获取响应
                api_response = self.api.get_health_advice(user_input)
                
                # 检查是否包含视力相关关键词
                if self.needs_vision_test(user_input):
                    choice = input("\n建议进行视力检测，是否现在开始？(y/n): ").lower()
                    if choice == 'y':
                        self.vision_result = self.run_vision_test()
                        print(f"视力检测结果已记录: {self.vision_result}")
                        
                        # 将视力结果整合到API调用中
                        combined_input = f"{user_input}。视力检测结果：{self.vision_result}"
                        api_response = self.api.get_health_advice(combined_input)
                
                # 显示当前建议
                print("\n助手建议:")
                print(api_response)
                
                # 判断是否需要继续对话（这里需要根据实际返回内容进行调整）
                if '建议尽快就医' in api_response:
                    print("\n[系统] 建议尽快就医，本次咨询结束")
                    break
                
                # 添加助手回复到对话历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": api_response
                })
                
            except Exception as e:
                print(f"处理出错: {str(e)}")
                break

    def _handle_image_input(self):
        """处理图片输入"""
        image_path = input("请输入图片路径：")
        img_response = self.handle_image_input(image_path)
        print(img_response)
        
        # 获取图片分析结果
        image_analysis = process_image(image_path)
        print(f"\n图片分析结果: {image_analysis}")
        
        # 获取补充症状描述
        symptoms = input("请补充描述您的症状（若无请直接回车）：")
        
        # 合并图片分析结果和文字症状
        combined_input = f"图片分析结果: {image_analysis}"
        if symptoms:
            combined_input += f"\n补充症状: {symptoms}"
        
        # 调用API获取建议
        advice = self.api.get_health_advice(combined_input)
        print("\n=== 综合诊断建议 ===")
        print(advice)

    def start_vision_game(self):
        """启动视力训练游戏"""
        print("正在启动视力训练游戏...")
        # 游戏启动逻辑
        pass

    def _display_structured_result(self, result):
        """显示结构化结果"""
        print("\n[详细分析]")
        print(f"可能病症: {', '.join(result.get('diagnosis', []))}")
        print(f"建议检查: {', '.join(result.get('examinations', []))}")
        print(f"护理建议: {', '.join(result.get('advice', []))}")
        print(f"紧急程度: {result.get('urgency', 3)}")

# --- 辅助函数 ---
def generate_report(user_id, symptoms, response):
    """生成报告文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{REPORT_DIR}/{user_id}_{timestamp}.txt"
    
    report_content = f"""=== 眼科健康报告 ===
用户ID: {user_id}
生成时间: {timestamp}
症状描述: {symptoms}
诊断建议:
{response}
"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    return filename

def ask_deepseek(question):
    """使用DeepSeek API进行问答"""
    return DeepseekAPI().get_health_advice(question)

def extract_keywords(text):
    """使用 DeepSeek API 提取关键词"""
    result = DeepseekAPI().get_health_advice(f"提取以下文本中的关键词（最多5个）：{text}")
    return result.get("keywords", [])

def run_vision_test():
    """启动视力检测并获取结果"""
    try:
        tester = VisionTester()
        result = tester.run_test()
        return result
    except Exception as e:
        print(f"视力检测失败: {str(e)}")
        return None

# --- Flask API服务 ---
@app.route('/api/process-input', methods=['POST'])
def process_input():
    data = request.json
    user_id = data.get('user_id')
    input_type = data.get('input_type')
    input_data = data.get('input_data')

    user_mem = get_user_memory(user_id)
    response = ""
    
    if input_type == 'text':
        if input_data.startswith("问:"):
            question = input_data[2:].strip()
            response = ask_deepseek(question)
        else:
            # 使用主API获取完整响应
            api_result = DeepseekAPI().get_health_advice(input_data)
            
            if "error" in api_result:
                response = f"系统错误: {api_result['error']}"
            else:
                response = api_result.get("diagnosis", "")
                
                # 处理特殊关键词功能
                keywords = api_result.get("keywords", [])
                if "眼保健操" in keywords:
                    response += "\n\n[眼保健操指导视频已准备]"
                elif "游戏" in keywords:
                    response += "\n\n[视力训练游戏已启动]"
                
                # 处理视力测试需求
                if any(kw in keywords for kw in ["看不清", "模糊", "视力下降"]):
                    try:
                        vision_score = run_vision_test()
                        response += f"\n\n视力检测结果: {vision_score:.1f}"
                    except Exception as e:
                        response += f"\n\n视力检测失败: {str(e)}"
            
            # 生成报告
            report_path = generate_report(user_id, input_data, response)
            app.logger.info(f"报告已生成: {report_path}")
            
    elif input_type == 'image':
        # 使用图片处理函数进行分析
        image_analysis = process_image(input_data)
        response = f"图片分析结果: {image_analysis}"
        report_path = generate_report(user_id, "图片分析", response)
        app.logger.info(f"图片分析报告已生成: {report_path}")

    update_user_memory(user_id, {"last_response": response})
    return jsonify({
        "response": response,
        "report_path": report_path,
        "keywords": []  # 图片模式仍然不返回关键词
    })

# --- 主程序入口 ---
def main():
    """简化版启动函数，仅支持命令行交互"""
    print("🖥️ 启动命令行交互界面")
    try:
        system = EyeHealthSystem()
        system.start()
    except Exception as e:
        print(f"系统启动失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 添加启动横幅
    print("""
    ███████╗██╗░░░██╗███████╗  ██╗░░██╗███████╗░█████╗░██╗░░██╗███████╗
    ██╔════╝╚██╗░██╔╝██╔════╝  ██║░░██║██╔════╝██╔══██╗██║░██╔╝██╔════╝
    █████╗░░░╚████╔╝░█████╗░░  ███████║█████╗░░███████║███████═╝█████╗░░
    ██╔══╝░░░░╚██╔╝░░██╔══╝░░  ██╔══██║██╔══╝░░██╔══██║██╔═██╗░██╔══╝░░
    ███████╗░░░██║░░░███████╗  ██║░░██║███████╗██║░░██║██║░╚██╗███████╗
    ╚══════╝░░░╚═╝░░░╚══════╝  ╚═╝░░╚═╝╚══════╝╚═╝░░╚═╝╚══════╝
    """)
    main()