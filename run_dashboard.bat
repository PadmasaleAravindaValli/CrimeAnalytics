@echo off
title Crime Analytics & ML Research Command Center
echo =====================================================================
echo  Starting Interactive Crime ML Research Dashboard...
echo  Comparative Evaluation of Machine Learning and Deep Learning
echo =====================================================================
cd /d "%~dp0"
"C:\Users\aru12\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
pause
