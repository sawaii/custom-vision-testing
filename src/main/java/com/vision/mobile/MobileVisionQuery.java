package com.vision.mobile;

import io.appium.java_client.AppiumDriver;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.javacpp.DoublePointer;
import org.openqa.selenium.OutputType;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;

public class MobileVisionQuery {
    private static final String TESS_DATA_PATH = "/opt/homebrew/share/tessdata";
    private final AppiumDriver driver;
    private final Tesseract tesseract;

    static {
        try {
            org.bytedeco.javacpp.Loader.load(org.bytedeco.opencv.global.opencv_core.class);
            System.out.println("OpenCV native library loaded successfully");
        } catch (Exception e) {
            System.err.println("Error loading OpenCV: " + e.getMessage());
            throw new RuntimeException("Failed to load OpenCV", e);
        }
    }

    public MobileVisionQuery(AppiumDriver driver) {
        this.driver = driver;
        this.tesseract = new Tesseract();
        tesseract.setDatapath(TESS_DATA_PATH);
        
        // Configure Tesseract for better text detection
        tesseract.setPageSegMode(3); // PSM_AUTO
        tesseract.setOcrEngineMode(1); // LSTM_ONLY
        
        // Set additional Tesseract parameters
        tesseract.setTessVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ");
        tesseract.setTessVariable("tessedit_create_hocr", "0");
        tesseract.setTessVariable("tessedit_create_txt", "1");
        tesseract.setTessVariable("tessedit_pageseg_mode", "3");
        tesseract.setTessVariable("tessedit_do_invert", "0");
        tesseract.setTessVariable("debug_file", "/dev/null");
        tesseract.setTessVariable("user_defined_dpi", "300");
    }

    public void saveCurrentScreen(String fileName) throws IOException {
        byte[] screenshotBytes = driver.getScreenshotAs(OutputType.BYTES);
        BufferedImage image = ImageIO.read(new ByteArrayInputStream(screenshotBytes));
        ImageIO.write(image, "png", new File(fileName));
    }

    public Mat getCurrentScreenAsMat() throws IOException {
        byte[] screenshotBytes = driver.getScreenshotAs(OutputType.BYTES);
        BufferedImage screenshot = ImageIO.read(new ByteArrayInputStream(screenshotBytes));
        return bufferedImageToMat(screenshot);
    }

    public Mat preprocessImage(Mat image) {
        Mat processed = new Mat();
        
        // Convert to grayscale if needed
        if (image.channels() > 1) {
            cvtColor(image, processed, COLOR_BGR2GRAY);
        } else {
            image.copyTo(processed);
        }
        
        // Apply bilateral filter to reduce noise while preserving edges
        Mat denoised = new Mat();
        bilateralFilter(processed, denoised, 3, 75, 75);
        processed.release();
        
        // Enhance contrast using CLAHE
        Mat equalized = new Mat();
        CLAHE clahe = createCLAHE(3.5, new Size(8, 8));
        clahe.apply(denoised, equalized);
        denoised.release();
        
        // Create sharpening kernel
        float[] kernelData = new float[] {
            -1.0f, -1.0f, -1.0f,
            -1.0f,  9.0f, -1.0f,
            -1.0f, -1.0f, -1.0f
        };
        Mat kernel = new Mat(3, 3, CV_32F);
        FloatBuffer kernelBuffer = kernel.createBuffer();
        for (float value : kernelData) {
            kernelBuffer.put(value);
        }
        
        // Apply sharpening
        Mat sharpened = new Mat();
        filter2D(equalized, sharpened, -1, kernel);
        equalized.release();
        kernel.release();
        
        // Apply Otsu's thresholding
        Mat binary = new Mat();
        threshold(sharpened, binary, 0, 255, THRESH_BINARY + THRESH_OTSU);
        sharpened.release();
        
        // Apply morphological operations
        Mat element = getStructuringElement(MORPH_RECT, new Size(3, 3));
        Mat cleaned = new Mat();
        morphologyEx(binary, cleaned, MORPH_OPEN, element);
        morphologyEx(cleaned, cleaned, MORPH_CLOSE, element);
        binary.release();
        element.release();
        
        return cleaned;
    }

    public String performOCR(Mat image, Rect region) {
        try {
            // Add padding to ensure full text capture
            int padding = 25;
            int x = Math.max(0, region.x() - padding);
            int y = Math.max(0, region.y() - padding);
            int width = Math.min(image.cols() - x, region.width() + 2 * padding);
            int height = Math.min(image.rows() - y, region.height() + 2 * padding);
            
            if (width <= 0 || height <= 0) {
                System.err.println("Invalid region dimensions after bounds checking");
                return "";
            }
            
            // Extract region with padding
            Rect adjustedRegion = new Rect(x, y, width, height);
            Mat roi = new Mat(image, adjustedRegion);
            
            // Scale up for better OCR
            Mat scaledRoi = new Mat();
            double scale = 5.0;
            resize(roi, scaledRoi, new Size(), scale, scale, INTER_CUBIC);
            roi.release();
            
            // Preprocess the region
            Mat processedRoi = preprocessImage(scaledRoi);
            scaledRoi.release();
            
            // Save processed image for debugging
            saveMatAsImage(processedRoi, "debug_" + x + "_" + y + ".png");
            
            // Convert to BufferedImage
            BufferedImage bufferedImage = matToBufferedImage(processedRoi);
            processedRoi.release();
            
            // Configure Tesseract for this specific region
            tesseract.setPageSegMode(7); // PSM_SINGLE_LINE
            tesseract.setTessVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 हिंदी");
            tesseract.setTessVariable("tessedit_do_invert", "0");
            tesseract.setTessVariable("debug_file", "/dev/null");
            tesseract.setTessVariable("tessedit_pageseg_mode", "7");
            tesseract.setTessVariable("tessedit_write_images", "1");
            tesseract.setTessVariable("textord_heavy_nr", "1");
            tesseract.setTessVariable("edges_max_children_per_outline", "50");
            tesseract.setTessVariable("edges_children_per_grandchild", "20");
            tesseract.setTessVariable("edges_children_count_limit", "50");
            tesseract.setTessVariable("edges_min_nonhole", "15");
            tesseract.setTessVariable("edges_max_nonhole", "3");
            tesseract.setTessVariable("textord_show_tables", "0");
            tesseract.setTessVariable("textord_tablefind_recognize_tables", "0");
            tesseract.setTessVariable("textord_tabfind_find_tables", "0");
            tesseract.setTessVariable("tessedit_enable_dict_correction", "1");
            tesseract.setTessVariable("tessedit_enable_bigram_correction", "1");
            tesseract.setTessVariable("tessedit_enable_fix_fuzzy_spaces", "1");
            tesseract.setTessVariable("tessedit_unrej_any_wd", "1");
            tesseract.setTessVariable("tessedit_fix_fuzzy_spaces", "1");
            tesseract.setTessVariable("tessedit_char_blacklist", "{}[]()@#$%^&*+=<>~`");
            tesseract.setTessVariable("tessedit_fix_hyphens", "1");
            tesseract.setTessVariable("tessedit_write_params_to_file", "");
            tesseract.setTessVariable("textord_force_make_prop_words", "F");
            tesseract.setTessVariable("textord_debug_block_rejection", "0");
            tesseract.setTessVariable("textord_min_linesize", "2.5");
            tesseract.setTessVariable("textord_debug_tabfind", "0");
            tesseract.setTessVariable("textord_show_initial_words", "0");
            tesseract.setTessVariable("textord_show_new_words", "0");
            tesseract.setTessVariable("textord_show_fixed_words", "0");
            
            // Perform OCR multiple times and use the most common result
            String result1 = tesseract.doOCR(bufferedImage).trim();
            String result2 = tesseract.doOCR(bufferedImage).trim();
            String result3 = tesseract.doOCR(bufferedImage).trim();
            
            // Clean up the results
            result1 = result1.replaceAll("\\s+", " ").trim();
            result2 = result2.replaceAll("\\s+", " ").trim();
            result3 = result3.replaceAll("\\s+", " ").trim();
            
            // Return the most common result
            if (result1.equals(result2) || result1.equals(result3)) {
                return result1;
            } else if (result2.equals(result3)) {
                return result2;
            }
            return result1;
        } catch (TesseractException e) {
            System.err.println("OCR failed: " + e.getMessage());
            e.printStackTrace();
            return "";
        }
    }

    private Mat bufferedImageToMat(BufferedImage image) {
        if (image.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            BufferedImage convertedImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
            convertedImage.getGraphics().drawImage(image, 0, 0, null);
            image = convertedImage;
        }

        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(image.getHeight(), image.getWidth(), CV_8UC3);
        ByteBuffer byteBuffer = mat.createBuffer();
        byteBuffer.put(pixels);
        
        return mat;
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int width = mat.cols();
        int height = mat.rows();
        int channels = mat.channels();
        byte[] data = new byte[width * height * channels];
        mat.data().get(data);
        
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        byte[] imageData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(data, 0, imageData, 0, data.length);
        
        return image;
    }

    public void saveMatAsImage(Mat mat, String fileName) {
        try {
            BufferedImage image = matToBufferedImage(mat);
            ImageIO.write(image, "png", new File(fileName));
        } catch (IOException e) {
            System.err.println("Failed to save Mat as image: " + e.getMessage());
        }
    }

    public boolean findLogoInRegion(Mat image, Rect region, Mat template) {
        try {
            // Extract region of interest
            Mat roi = new Mat(image, region);
            
            // Convert both images to grayscale
            Mat grayRoi = new Mat();
            Mat grayTemplate = new Mat();
            if (roi.channels() > 1) {
                cvtColor(roi, grayRoi, COLOR_BGR2GRAY);
            } else {
                roi.copyTo(grayRoi);
            }
            if (template.channels() > 1) {
                cvtColor(template, grayTemplate, COLOR_BGR2GRAY);
            } else {
                template.copyTo(grayTemplate);
            }
            
            // Resize template to match ROI dimensions
            Mat resizedTemplate = new Mat();
            resize(grayTemplate, resizedTemplate, new Size(region.width(), region.height()));
            
            // Create result matrix
            Mat result = new Mat();
            matchTemplate(grayRoi, resizedTemplate, result, TM_CCOEFF_NORMED);
            
            // Find best match
            DoublePointer minVal = new DoublePointer(1);
            DoublePointer maxVal = new DoublePointer(1);
            Point minLoc = new Point();
            Point maxLoc = new Point();
            minMaxLoc(result, minVal, maxVal, minLoc, maxLoc, null);
            
            // Get the match confidence
            double matchConfidence = maxVal.get();
            
            // Clean up
            roi.release();
            grayRoi.release();
            grayTemplate.release();
            resizedTemplate.release();
            result.release();
            minVal.deallocate();
            maxVal.deallocate();
            
            // Return true if match confidence is above threshold
            return matchConfidence > 0.7;
        } catch (Exception e) {
            System.err.println("Error in template matching: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
} 