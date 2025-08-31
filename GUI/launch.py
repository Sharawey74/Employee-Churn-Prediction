#!/usr/bin/env python3
"""
Launch script for Employee Churn Prediction GUI
Checks dependencies and starts the Streamlit application
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'plotly', 'pandas', 'numpy', 'sklearn', 'xgboost', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install requirements from requirements.txt"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        return True
    except subprocess.CalledProcessError:
        return False

def launch_app():
    """Launch the Streamlit application"""
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"Error launching application: {e}")

def main():
    """Main launcher function"""
    print("🚀 Employee Churn Prediction GUI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Error: app.py not found!")
        print("Please run this script from the GUI directory.")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"📦 Missing packages: {', '.join(missing)}")
        print("🔧 Installing requirements...")
        
        if install_requirements():
            print("✅ Requirements installed successfully!")
        else:
            print("❌ Failed to install requirements.")
            print("Please install manually: pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("✅ All dependencies found!")
    
    # Launch application
    print("\n🌐 Launching Streamlit application...")
    print("📍 URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application\n")
    
    launch_app()

if __name__ == "__main__":
    main()
