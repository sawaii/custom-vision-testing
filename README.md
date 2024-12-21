# Mobile Vision Testing Framework

A Java-based mobile testing framework that combines Appium with OpenCV and Tesseract OCR for robust visual testing and text recognition in mobile applications.

## Features

- Automated mobile app testing using Appium
- Image processing and template matching with OpenCV
- Text recognition using Tesseract OCR
- Support for multiple languages including Hindi
- Configurable image preprocessing for better OCR accuracy
- Detailed logging and debugging capabilities

## Prerequisites

- Java JDK 18 or higher
- Maven
- Appium Server 2.0 or higher
- Android SDK (for Android testing)
- Tesseract OCR with language data
- OpenCV (installed via Maven dependency)

## Setup

1. Install Tesseract OCR:
```bash
brew install tesseract
```

2. Install language data for Hindi:
```bash
brew install tesseract-lang
```

3. Clone the repository:
```bash
git clone https://github.com/sawaii/custom-vision-testing.git
cd custom-vision-testing
```

4. Install dependencies:
```bash
mvn clean install
```

## Configuration

1. Update `src/test/resources/config.properties` with your Appium settings:
```properties
appium.server.url=http://127.0.0.1:4723
appium.device.name=your_device_name
appium.app.path=/path/to/your/app
appium.automation.name=UiAutomator2
appium.no.reset=true
appium.full.reset=false
appium.new.command.timeout=60
appium.auto.grant.permissions=true
```

## Running Tests

Run tests using Maven:
```bash
mvn clean test
```

Run a specific test:
```bash
mvn clean test -Dtest=MobileVisionTest#testLanguageSelectionPage
```

## Project Structure

- `src/main/java/com/vision/mobile/` - Core framework classes
  - `MobileVisionQuery.java` - Main class for image processing and OCR
- `src/test/java/com/vision/test/` - Test classes
  - `MobileVisionTest.java` - Example test implementation
- `src/test/resources/` - Test resources and configuration files

## Features in Detail

### Image Processing
- Bilateral filtering for noise reduction
- CLAHE for contrast enhancement
- Custom sharpening kernel
- Otsu's thresholding
- Morphological operations

### OCR Capabilities
- Multi-language text recognition
- Region-based text extraction
- Multiple OCR attempts for accuracy
- Configurable preprocessing parameters

### Template Matching
- Logo detection
- UI element verification
- Confidence threshold adjustment

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.