#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Custom Vision Testing Framework Setup...${NC}"

# Detect OS
OS="$(uname)"
echo -e "Detected OS: ${GREEN}$OS${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Java version
check_java() {
    if command_exists java; then
        java_version=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
        if [ "$java_version" -ge 11 ]; then
            echo -e "${GREEN}✓ Java $java_version installed${NC}"
            return 0
        else
            echo -e "${RED}✗ Java version must be 11 or higher (found version $java_version)${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Java not found${NC}"
        return 1
    fi
}

# Function to check Maven
check_maven() {
    if command_exists mvn; then
        echo -e "${GREEN}✓ Maven installed${NC}"
        return 0
    else
        echo -e "${RED}✗ Maven not found${NC}"
        return 1
    fi
}

# Function to check/install Tesseract
setup_tesseract() {
    if command_exists tesseract; then
        echo -e "${GREEN}✓ Tesseract already installed${NC}"
    else
        echo -e "${YELLOW}Installing Tesseract...${NC}"
        case $OS in
            "Darwin")  # macOS
                brew install tesseract tesseract-lang
                ;;
            "Linux")
                sudo apt-get update
                sudo apt-get install -y tesseract-ocr tesseract-ocr-all
                ;;
            *)
                echo -e "${RED}Please install Tesseract manually for your OS${NC}"
                return 1
                ;;
        esac
    fi
}

# Function to check/install OpenCV
setup_opencv() {
    case $OS in
        "Darwin")  # macOS
            if brew list opencv &>/dev/null; then
                echo -e "${GREEN}✓ OpenCV already installed${NC}"
            else
                echo -e "${YELLOW}Installing OpenCV...${NC}"
                brew install opencv
            fi
            ;;
        "Linux")
            if dpkg -l | grep -q libopencv; then
                echo -e "${GREEN}✓ OpenCV already installed${NC}"
            else
                echo -e "${YELLOW}Installing OpenCV...${NC}"
                sudo apt-get update
                sudo apt-get install -y libopencv-dev
            fi
            ;;
        *)
            echo -e "${RED}Please install OpenCV manually for your OS${NC}"
            return 1
            ;;
    esac
}

# Function to check/install Node.js and Appium
setup_appium() {
    if ! command_exists node; then
        echo -e "${RED}Node.js not found. Please install Node.js first${NC}"
        return 1
    fi

    if command_exists appium; then
        echo -e "${GREEN}✓ Appium already installed${NC}"
    else
        echo -e "${YELLOW}Installing Appium...${NC}"
        npm install -g appium
        npm install -g appium-doctor
    fi
}

# Function to create project structure
setup_project_structure() {
    echo -e "${YELLOW}Creating project structure...${NC}"
    
    # Create templates directory
    mkdir -p src/test/resources/templates
    
    # Create logs directory
    mkdir -p logs
    
    echo -e "${GREEN}✓ Project structure created${NC}"
}

# Main setup process
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

# Check Java
check_java || exit 1

# Check Maven
check_maven || exit 1

# Setup Tesseract
echo -e "\n${YELLOW}Setting up Tesseract...${NC}"
setup_tesseract || exit 1

# Setup OpenCV
echo -e "\n${YELLOW}Setting up OpenCV...${NC}"
setup_opencv || exit 1

# Setup Appium
echo -e "\n${YELLOW}Setting up Appium...${NC}"
setup_appium || exit 1

# Setup project structure
echo -e "\n${YELLOW}Setting up project structure...${NC}"
setup_project_structure

# Install Maven dependencies
echo -e "\n${YELLOW}Installing Maven dependencies...${NC}"
mvn clean install

echo -e "\n${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update Tesseract data path in VisionQuery.java"
echo "2. Configure Appium capabilities in MobileVisionTest.java"
echo "3. Add template images to src/test/resources/templates/"
echo "4. Run 'mvn test' to verify setup" 