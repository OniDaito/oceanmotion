package oceanmotion;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import nom.tam.fits.Fits;
import nom.tam.fits.ImageHDU;
import java.io.File;
import java.util.Vector;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.RescaleOp;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.io.InputStream;
import java.io.FileInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


public class ReadFITS {
    private static final Logger logger = Logger.getLogger(ReadFITS.class.getName());

    public static byte[] uncompressLz4File(String filePath) throws java.io.IOException, java.lang.InterruptedException {
        // Build the lz4 command
        ProcessBuilder processBuilder = new ProcessBuilder("lz4", "-d", "-f", filePath, "temp.fits");

        // Start the process
        Process process = processBuilder.start();

        // Wait for the process to complete
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.err.println(line);
                }
            }
            throw new IOException("Decompression failed with exit code " + exitCode);
        }

        // Read the decompressed file back into a byte array
        Path path = Paths.get("temp.fits");
        return Files.readAllBytes(path);

    }

    public static Vector<String> read_dir(String directoryPath) {
        // LZ4 Compresed FITS files but just from sonar 854.
        String extension = "854.fits.lz4";

        File folder = new File(directoryPath);
        Vector<String> filePaths = new Vector<>();

        // Get the list of files in the directory
        File[] files = folder.listFiles();

        // Loop through each file and check the extension
        for (File file : files) {
            if (file.isFile() && file.getName().endsWith(extension)) {
                // File has the desired extension, add the path to the vector
                filePaths.add(file.getAbsolutePath());
            }
        }

        return filePaths;
    }

    public static byte[][] read(String fitsFilePath) throws java.io.IOException, java.lang.InterruptedException {
        logger.info("Reading: " + fitsFilePath);
        byte[] uncompressed = uncompressLz4File(fitsFilePath);
        
        InputStream targetStream = new ByteArrayInputStream(uncompressed);

        // New FITS file from the decompressed bytes.
        Fits fitsFile = new Fits("temp.fits");
        
        ImageHDU hdu = (ImageHDU) fitsFile.readHDU();
        // Should be uint8 but Java is silly.
        byte[][] image = (byte[][]) hdu.getKernel();
                
        // Close the FITSFile object
        fitsFile.close();
       
        return image;
    }

    public static float[][] prepare_fits(byte[][] raw_fits) {
        // Prepare the fits file by performing an initial crop, then a resize to
        // match the dimensions that the neural network expects.
        // Then, perform a conversion to float in the range 0 to 1.

        // Convert float[][] to BufferedImage
        int width = raw_fits[0].length;
        int height = raw_fits.length;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Raster raster = image.getRaster();
        DataBufferByte buffer = (DataBufferByte) raster.getDataBuffer();
        byte[] data = buffer.getData();

        // Copy data from float[][] to BufferedImage
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                data[i * width + j] = raw_fits[i][j];
            }
        }

        // Crop the image
        int cropX = 0;
        int cropY = 0;
        int cropWidth = 512;
        int cropHeight = 1632;
        BufferedImage croppedImage = image.getSubimage(cropX, cropY, cropWidth, cropHeight);

        // Resize the image
        int newWidth = 256;
        int newHeight = 816;
        BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resizedImage.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(croppedImage, 0, 0, newWidth, newHeight, null);
        g.dispose();

        float[][] resizedImageData = new float[newHeight][newWidth];
        raster = resizedImage.getRaster();

        // Copy back
        buffer = (DataBufferByte) raster.getDataBuffer();
        data = buffer.getData();

        // Copy data from float[][] to BufferedImage
        for (int i = 0; i < newHeight; i++) {
            for (int j = 0; j < newWidth; j++) {
                int d = data[i * newWidth + j];
                resizedImageData[i][j] = (float) d / (float)255.0;
            }
        }

        return resizedImageData;
    }

    
}