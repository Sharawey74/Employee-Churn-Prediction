#!/usr/bin/env python3
"""
Path Navigation Demo - Understanding .parent.parent
"""

from pathlib import Path

def demo_path_navigation():
    print("🧭 Path Navigation Demo")
    print("="*50)
    
    # Simulate being in main/main.py
    current_file = Path("C:/Users/DELL/Desktop/AI-Project/AI-Project/main/main.py")
    
    print(f"📍 Current file: {current_file}")
    print(f"📁 Current folder: {current_file.parent}")
    print(f"🏠 Project root: {current_file.parent.parent}")
    print(f"🎯 src folder: {current_file.parent.parent / 'src'}")
    
    print("\n🔧 Step by step:")
    print(f"1. __file__ = {current_file}")
    print(f"2. Path(__file__) = {current_file}")  
    print(f"3. Path(__file__).parent = {current_file.parent}")
    print(f"4. Path(__file__).parent.parent = {current_file.parent.parent}")
    
    print("\n📊 Why we need .parent.parent:")
    print("   main.py is in: main/")
    print("   src/ is in:    project_root/")
    print("   So we go: main/ → project_root/ → src/")
    print("             ^1st parent  ^2nd parent")

if __name__ == "__main__":
    demo_path_navigation()
