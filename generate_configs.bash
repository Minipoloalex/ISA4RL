cd src/main/
uv run --env-file .env ../agents_highway/build_configs.py > ../../config/logs_he.txt

cd ../../
cd src/agents_metadrive/
uv run build_configs.py > ../../config/logs_md.txt
