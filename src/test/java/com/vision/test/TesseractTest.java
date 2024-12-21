package com.vision.test;

import com.vision.core.VisionQuery;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

public class TesseractTest {
    private VisionQuery visionQuery;

    @BeforeClass
    public void setup() {
        visionQuery = new VisionQuery();
    }

    @Test
    public void testTesseractInitialization() throws Exception {
        // Create a simple test image with text
        BufferedImage image = new BufferedImage(200, 50, BufferedImage.TYPE_INT_RGB);
        Graphics g = image.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 200, 50);
        g.setColor(Color.BLACK);
        g.setFont(new Font("Arial", Font.PLAIN, 24));
        g.drawString("Hello World", 10, 30);
        g.dispose();

        // Save the test image
        File testImage = new File("test_image.png");
        ImageIO.write(image, "PNG", testImage);

        // Try to read text from the image
        String result = visionQuery.findText(image);
        System.out.println("OCR Result: " + result);

        // Verify that some text was found
        Assert.assertNotNull(result, "OCR result should not be null");
        Assert.assertTrue(result.trim().length() > 0, "OCR result should not be empty");

        // Clean up
        testImage.delete();
    }
} 