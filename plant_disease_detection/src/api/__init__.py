# src/api/__init__.py

from flask import Flask
from flask_cors import CORS
import os

def create_app():
    """创建并配置Flask应用"""
    app = Flask(__name__, 
                static_folder='../../static', 
                template_folder='../../templates')
    
    # 启用CORS支持
    CORS(app)
    
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    
    # 注册API路由蓝图
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # 注册前端路由
    from .views import views_bp
    app.register_blueprint(views_bp)
    
    return app