package com.vision.test;

import com.vision.mobile.MobileVisionQuery;
import io.appium.java_client.AppiumDriver;
import io.appium.java_client.android.AndroidDriver;
import io.appium.java_client.android.options.UiAutomator2Options;
import org.bytedeco.opencv.opencv_core.*;
import org.openqa.selenium.Dimension;
import org.testng.annotations.*;
import org.testng.Assert;

import java.io.FileInputStream;
import java.io.IOException;
import java.net.URL;
import java.time.Duration;
import java.util.Properties;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;

public class MobileVisionTest {
    private AppiumDriver driver;
    private MobileVisionQuery visionQuery;
    private Properties config;

    @BeforeClass
    public void setup() throws Exception {
        // Load configuration
        config = new Properties();
        config.load(new FileInputStream("src/test/resources/config.properties"));

        // Set up UiAutomator2Options for Appium 2.0
        UiAutomator2Options options = new UiAutomator2Options()
            .setDeviceName(config.getProperty("appium.device.name"))
            .setApp(config.getProperty("appium.app.path"))
            .setAutomationName(config.getProperty("appium.automation.name"))
            .setNoReset(Boolean.parseBoolean(config.getProperty("appium.no.reset")))
            .setFullReset(Boolean.parseBoolean(config.getProperty("appium.full.reset")))
            .setNewCommandTimeout(Duration.ofSeconds(Integer.parseInt(config.getProperty("appium.new.command.timeout"))))
            .setAutoGrantPermissions(Boolean.parseBoolean(config.getProperty("appium.auto.grant.permissions")));
        
        // Initialize the driver with updated capabilities
        URL serverUrl = new URL(config.getProperty("appium.server.url"));
        driver = new AndroidDriver(serverUrl, options);
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
        
        // Initialize vision query
        visionQuery = new MobileVisionQuery(driver);
    }

    @Test
    public void testLanguageSelectionPage() throws IOException, InterruptedException {
        // Wait for the app to load
        Thread.sleep(10000);

        // Take a screenshot and save it
        visionQuery.saveCurrentScreen("language_selection_screen.png");

        // Get the screen dimensions
        Dimension screenSize = driver.manage().window().getSize();
        int screenWidth = screenSize.getWidth();
        int screenHeight = screenSize.getHeight();

        // Get the current screenshot as Mat
        Mat screenshot = visionQuery.getCurrentScreenAsMat();
        
        // Log image dimensions for debugging
        System.out.println("Image dimensions: " + screenshot.cols() + "x" + screenshot.rows());
        
        // Ensure regions are within image bounds
        int maxWidth = screenshot.cols();
        int maxHeight = screenshot.rows();
        
        // Create rectangles using ByteDeco's OpenCV API with bounds checking
        Rect headerRegion = new Rect(
            maxWidth / 4,
            maxHeight / 20,
            maxWidth / 2,
            maxHeight / 5
        ).position(0);

        Rect chooseLanguageRegion = new Rect(
            maxWidth / 4,
            maxHeight / 3,
            maxWidth / 2,
            maxHeight / 5
        ).position(0);

        Rect hindiRegion = new Rect(
            maxWidth / 4,
            maxHeight / 2,
            maxWidth / 2,
            maxHeight / 4
        ).position(0);

        Rect englishRegion = new Rect(
            maxWidth / 4,
            3 * maxHeight / 4 - maxHeight / 8,
            maxWidth / 2,
            maxHeight / 4
        ).position(0);

        // Log region dimensions for debugging
        System.out.println("Image dimensions: " + maxWidth + "x" + maxHeight);
        System.out.println("Header region: x=" + headerRegion.x() + ", y=" + headerRegion.y() + 
                         ", width=" + headerRegion.width() + ", height=" + headerRegion.height());
        System.out.println("Choose language region: x=" + chooseLanguageRegion.x() + ", y=" + chooseLanguageRegion.y() + 
                         ", width=" + chooseLanguageRegion.width() + ", height=" + chooseLanguageRegion.height());
        System.out.println("Hindi region: x=" + hindiRegion.x() + ", y=" + hindiRegion.y() + 
                         ", width=" + hindiRegion.width() + ", height=" + hindiRegion.height());
        System.out.println("English region: x=" + englishRegion.x() + ", y=" + englishRegion.y() + 
                         ", width=" + englishRegion.width() + ", height=" + englishRegion.height());
        
        // Save the preprocessed image for debugging
        visionQuery.saveCurrentScreen("original_screenshot.png");
        
        // Load the logo template
        Mat logoTemplate = imread("src/test/resources/templates/initial_screen.png", IMREAD_COLOR);
        if (logoTemplate.empty()) {
            throw new IOException("Failed to load logo template");
        }
        
        // Print template dimensions
        System.out.println("Template dimensions: " + logoTemplate.cols() + "x" + logoTemplate.rows());
        
        // Check for logo in header region
        boolean logoFound = visionQuery.findLogoInRegion(screenshot, headerRegion, logoTemplate);
        Assert.assertTrue(logoFound, "Header should contain Zupee logo");
        
        // Preprocess the image
        Mat processedImage = visionQuery.preprocessImage(screenshot);
        
        // Save the preprocessed image for debugging
        visionQuery.saveMatAsImage(processedImage, "preprocessed_image.png");
        
        // Perform OCR on each region
        String headerText = visionQuery.performOCR(processedImage, headerRegion);
        String chooseLanguageText = visionQuery.performOCR(processedImage, chooseLanguageRegion);
        String hindiText = visionQuery.performOCR(processedImage, hindiRegion);
        String englishText = visionQuery.performOCR(processedImage, englishRegion);
        
        // Log OCR results for debugging
        System.out.println("Header text: " + headerText);
        System.out.println("Choose language text: " + chooseLanguageText);
        System.out.println("Hindi text: " + hindiText);
        System.out.println("English text: " + englishText);
        
        // Log region objects for debugging
        System.out.println("Header region: " + headerRegion);
        System.out.println("Choose language region: " + chooseLanguageRegion);
        System.out.println("Hindi region: " + hindiRegion);
        System.out.println("English region: " + englishRegion);
        
        // Verify text presence with more flexible matching
        Assert.assertTrue(
            chooseLanguageText.toLowerCase().replaceAll("[^a-z\\s]", "").contains("choose") || 
            chooseLanguageText.toLowerCase().replaceAll("[^a-z\\s]", "").contains("language") ||
            chooseLanguageText.toLowerCase().replaceAll("[^a-z\\s]", "").contains("select") ||
            chooseLanguageText.toLowerCase().replaceAll("[^a-z\\s]", "").contains("lang"),
            "Choose language text should be present"
        );
        Assert.assertTrue(
            hindiText.toLowerCase().contains("hindi") || 
            hindiText.toLowerCase().contains("हिंदी") ||
            hindiText.toLowerCase().contains("हिन्दी"),
            "Hindi text should be present"
        );
        Assert.assertTrue(
            englishText.toLowerCase().contains("english") || 
            englishText.toLowerCase().contains("eng") ||
            englishText.toLowerCase().contains("en"),
            "English text should be present"
        );
        
        // Clean up
        screenshot.release();
        processedImage.release();
        logoTemplate.release();
    }

    @AfterClass
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
} 