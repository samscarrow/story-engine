@echo off
echo Starting KoboldCpp with DeepSeek-R1 model...
echo ==========================================
echo.
echo Model: DeepSeek-R1-0528-Qwen3-8B-Q4_K_M
echo Context: 4096 tokens
echo Port: 5001
echo.

koboldcpp.exe --model "/c/Users/sscar/.lmstudio/models/lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf" --contextsize 4096 --port 5001

echo.
echo KoboldCpp has stopped.
pause