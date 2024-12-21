package com.vision.core;

import lombok.extern.slf4j.Slf4j;
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import org.bytedeco.javacpp.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

@Slf4j
public class VisionQuery {
    private final ITesseract tesseract;

    public VisionQuery() {
        // Set system properties for JNA
        String libraryPath = "/System/Volumes/Data/opt/homebrew/lib";
        String leptonicaPath = "/System/Volumes/Data/opt/homebrew/Cellar/tesseract/5.5.0/lib";
        
        // Set library paths
        System.setProperty("jna.library.path", libraryPath + ":" + leptonicaPath);
        System.setProperty("jna.platform.library.path", libraryPath + ":" + leptonicaPath);
        System.setProperty("java.library.path", libraryPath + ":" + leptonicaPath);
        
        // Additional JNA settings
        System.setProperty("jna.encoding", "UTF8");
        System.setProperty("jna.platform", "darwin-aarch64");
        System.setProperty("jna.debug_load", "true");

        // Initialize Tesseract
        tesseract = new Tesseract();
        tesseract.setDatapath("/opt/homebrew/share/tessdata");
        tesseract.setLanguage("eng");
        tesseract.setOcrEngineMode(1); // LSTM_ONLY mode
        tesseract.setPageSegMode(3); // PSM_AUTO

        // Log system information for debugging
        log.info("System architecture: {}", System.getProperty("os.arch"));
        log.info("JNA library path: {}", System.getProperty("jna.library.path"));
        log.info("JNA platform library path: {}", System.getProperty("jna.platform.library.path"));
        log.info("JNA platform: {}", System.getProperty("jna.platform"));
        log.info("Java library path: {}", System.getProperty("java.library.path"));

        // Ensure Tesseract data path includes Hindi language data
        tesseract.setDatapath("/opt/homebrew/share/tessdata"); // Confirm this path contains 'hin.traineddata'
    }

    public String findText(File imageFile) throws TesseractException {
        if (!imageFile.exists()) {
            throw new IllegalArgumentException("Image file does not exist: " + imageFile.getAbsolutePath());
        }
        return tesseract.doOCR(imageFile);
    }

    public String findText(BufferedImage image) throws TesseractException {
        if (image == null) {
            throw new IllegalArgumentException("Image cannot be null");
        }
        return tesseract.doOCR(image);
    }

    public String findTextInRegion(BufferedImage image, int x, int y, int width, int height) throws TesseractException {
        BufferedImage regionImage = image.getSubimage(x, y, width, height);
        return findText(regionImage);
    }

    public List<Point2d> findElement(Mat screen, Mat template, double threshold) {
        List<Point2d> matches = new ArrayList<>();
        Mat result = new Mat();

        // Convert both images to grayscale
        Mat grayScreen = new Mat();
        Mat grayTemplate = new Mat();
        cvtColor(screen, grayScreen, COLOR_BGR2GRAY);
        cvtColor(template, grayTemplate, COLOR_BGR2GRAY);

        // Ensure both images are CV_8U type
        Mat screen8U = new Mat();
        Mat template8U = new Mat();
        grayScreen.convertTo(screen8U, CV_8U);
        grayTemplate.convertTo(template8U, CV_8U);

        matchTemplate(screen8U, template8U, result, TM_CCOEFF_NORMED);

        FloatPointer pointer = new FloatPointer(result.data());
        for (int i = 0; i < result.rows(); i++) {
            for (int j = 0; j < result.cols(); j++) {
                float value = pointer.get(i * result.cols() + j);
                if (value >= threshold) {
                    matches.add(new Point2d(j, i));
                }
            }
        }

        return matches;
    }

    public BufferedImage preprocessImage(BufferedImage original) {
        try {
            Mat mat = bufferedImageToMat(original);
            Mat gray = new Mat();
            cvtColor(mat, gray, COLOR_BGR2GRAY);
            Mat binary = new Mat();
            threshold(gray, binary, 0, 255, THRESH_BINARY + THRESH_OTSU);
            return matToBufferedImage(binary);
        } catch (IOException e) {
            log.error("Error preprocessing image: ", e);
            return original;
        }
    }

    public List<Rect> findElementsByColor(Mat image, Scalar lowerBound, Scalar upperBound) {
        List<Rect> elements = new ArrayList<>();
        
        Mat hsv = new Mat();
        cvtColor(image, hsv, COLOR_BGR2HSV);
        
        Mat mask = new Mat();
        Mat lowerMat = new Mat(1, 1, CV_32FC3, lowerBound);
        Mat upperMat = new Mat(1, 1, CV_32FC3, upperBound);
        inRange(hsv, lowerMat, upperMat, mask);
        
        MatVector contours = new MatVector();
        Mat hierarchy = new Mat();
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        long size = contours.size();
        for (long i = 0; i < size; i++) {
            Rect rect = boundingRect(contours.get(i));
            elements.add(rect);
        }
        
        return elements;
    }

    public double compareImages(Mat image1, Mat image2) {
        Size size1 = image1.size();
        Size size2 = image2.size();
        if (size1.width() != size2.width() || size1.height() != size2.height()) {
            throw new IllegalArgumentException("Images must be the same size");
        }

        Mat diff = new Mat();
        absdiff(image1, image2, diff);
        Mat gray = new Mat();
        cvtColor(diff, gray, COLOR_BGR2GRAY);
        Mat binary = new Mat();
        threshold(gray, binary, 30, 255, THRESH_BINARY);

        int nonZero = countNonZero(binary);
        int total = binary.rows() * binary.cols();
        
        return 1.0 - ((double) nonZero / total);
    }

    public void saveDebugImage(Mat image, String filename) {
        imwrite(filename, image);
    }

    private Mat bufferedImageToMat(BufferedImage image) throws IOException {
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(image.getHeight(), image.getWidth(), CV_8UC3);
        mat.data().put(pixels);
        return mat;
    }

    private BufferedImage matToBufferedImage(Mat mat) throws IOException {
        int width = mat.cols();
        int height = mat.rows();
        int channels = mat.channels();
        byte[] data = new byte[width * height * channels];
        mat.data().get(data);
        
        BufferedImage image = new BufferedImage(width, height, 
            channels == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR);
        
        if (channels == 1) {
            // Grayscale image
            image.getRaster().setDataElements(0, 0, width, height, data);
        } else {
            // Color image
            byte[] bgr = new byte[width * height * 3];
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int pixelIndex = (i * width + j) * channels;
                    int bgrIndex = (i * width + j) * 3;
                    for (int k = 0; k < Math.min(channels, 3); k++) {
                        bgr[bgrIndex + k] = data[pixelIndex + k];
                    }
                }
            }
            image.getRaster().setDataElements(0, 0, width, height, bgr);
        }
        
        return image;
    }
} 