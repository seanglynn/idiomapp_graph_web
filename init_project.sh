#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Initializing IdiomApp project structure...${NC}"

# Create necessary directories
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p ollama-models
mkdir -p logs

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
  echo -e "${YELLOW}Creating .env file from template...${NC}"
  cp .env.example .env
  echo -e "${GREEN}Created .env file. Please edit it with your configuration.${NC}"
else
  echo -e "${YELLOW}.env file already exists, skipping...${NC}"
fi

# Make the logs directory writable
chmod -R 777 logs

# Create Docker volume directories
echo -e "${YELLOW}Setting up Docker volume directories...${NC}"
mkdir -p ollama-models
chmod -R 777 ollama-models

echo -e "${GREEN}Project initialization complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Edit .env file with your configuration"
echo -e "2. Run 'make docker-start' to start the application"
echo -e "3. Run 'make docker-pull-model' to download the LLM model (first time only)"
echo -e "4. Access the application at http://localhost:8503" 