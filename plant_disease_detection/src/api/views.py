# src/api/views.py

from flask import Blueprint, render_template, send_from_directory
import os

# 创建视图蓝图
views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@views_bp.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)