package com.vision.test;

import com.vision.mobile.MobileVisionQuery;
import io.appium.java_client.AppiumDriver;
import io.appium.java_client.android.AndroidDriver;
import org.opencv.core.*;
import org.testng.annotations.*;
import org.openqa.selenium.remote.DesiredCapabilities;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URL;
import java.time.Duration;
import java.util.Properties;

public class TemplateCaptureUtil {
    private AppiumDriver driver;
    private MobileVisionQuery visionQuery;
    private Properties config;

    @BeforeClass
    public void setup() throws Exception {
        // Load configuration
        config = new Properties();
        config.load(new FileInputStream("src/test/resources/config.properties"));

        DesiredCapabilities caps = new DesiredCapabilities();
        caps.setCapability("platformName", config.getProperty("appium.platform.name"));
        caps.setCapability("deviceName", config.getProperty("appium.device.name"));
        caps.setCapability("automationName", config.getProperty("appium.automation.name"));
        caps.setCapability("app", config.getProperty("appium.app.path"));
        caps.setCapability("noReset", Boolean.parseBoolean(config.getProperty("appium.no.reset")));
        caps.setCapability("fullReset", Boolean.parseBoolean(config.getProperty("appium.full.reset")));
        caps.setCapability("newCommandTimeout", Integer.parseInt(config.getProperty("appium.new.command.timeout")));
        caps.setCapability("autoGrantPermissions", Boolean.parseBoolean(config.getProperty("appium.auto.grant.permissions")));
        
        // Initialize the driver with updated capabilities
        URL serverUrl = new URL(config.getProperty("appium.server.url"));
        driver = new AndroidDriver(serverUrl, caps);
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
        
        // Initialize vision query
        visionQuery = new MobileVisionQuery(driver);
    }

    @Test
    public void captureTemplate() throws Exception {
        // Wait for app to load
        Thread.sleep(5000);

        // Take a screenshot and save it
        visionQuery.saveCurrentScreen("template_screen.png");
    }

    @AfterClass
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
} 