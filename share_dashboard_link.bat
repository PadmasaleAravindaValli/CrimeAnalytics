@echo off
title Shareable Public Link - Crime ML Research Dashboard
echo =====================================================================
echo  Starting Dashboard and Generating Public Shareable Web Link...
echo =====================================================================
cd /d "%~dp0"
start "" /b "C:\Users\aru12\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py --server.headless=true
timeout /t 4 /nobreak > nul
echo.
echo =====================================================================
echo  LOOK BELOW FOR YOUR LIVE PUBLIC LINK (trycloudflare.com):
echo  Copy that link and send it on WhatsApp/Email!
echo  Anyone can open it on their phone or laptop with NO code or login!
echo =====================================================================
echo.
"cloudflared.exe" tunnel --url http://localhost:8501 --protocol http2
pause
