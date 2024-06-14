package oceanmotion;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;

import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Vector;
import java.util.List;
import java.util.Collections;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.io.File;
import javax.imageio.ImageIO;

import org.bytedeco.ffmpeg.avcodec.AVCodecContext;
import org.bytedeco.ffmpeg.avcodec.AVPacket;
import org.bytedeco.ffmpeg.avutil.AVRational;
import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.ffmpeg.global.swscale;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;

import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imencode;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_GRAY2BGR;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

// Useful links
// https://github.com/microsoft/onnxruntime/blob/main/java/src/test/java/sample/ScoreMNIST.java

/**
 * Load some FITS files the test dataset and make
 * a prediction, cropping and resizing.
 * 
 * Assumes args[0] is the path to the model.onnx file.
 * 
 * Example usage under MAVEN - mvn exec:java -Dexec.mainClass="oceanmotion.WrapperProg" -Dexec.args="/home/oni/Projects/oceanmotion/model.onnx /home/oni/Projects/sealhits_testdata/fits/2023_05_28"
 * 
 */
public class WrapperProg  {
    private static final Logger logger = Logger.getLogger(WrapperProg.class.getName());

    private static final int WINDOW_LENGTH = 16;
    private static final int IMG_HEIGHT = 816;
    private static final int IMG_WIDTH = 256;
    private static final float CONFIDENCE = 0.8f;

    private static BufferedImage convertToBufferedImage(float[][] frame) {
        int height = frame.length;
        int width = frame[0].length;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int intensity = (int) (frame[y][x] * 255);
                int rgb = (intensity << 16) | (intensity << 8) | intensity;
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }

    private static Mat bufferedImageToMat(BufferedImage bi) {
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), org.bytedeco.opencv.global.opencv_core.CV_8UC1);
        byte[] data = ((java.awt.image.DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        mat.data().put(data);
        return mat;
    }

    public static void convertToWebM(Vector<BufferedImage> frames, String outputPath) {
        if (frames.isEmpty()) {
            throw new IllegalArgumentException("The frames vector is empty.");
        }

        int width = frames.get(0).getWidth();
        int height = frames.get(0).getHeight();
        int frameRate = 4; // 4 FPS

        try (FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(outputPath, width, height)) {
                recorder.setVideoCodec(avcodec.AV_CODEC_ID_VP8);
                recorder.setFormat("webm");
                recorder.setFrameRate(frameRate);
                recorder.setVideoOption("crf", "18");
                recorder.setPixelFormat(avutil.AV_PIX_FMT_YUV420P);
                recorder.start();
                
                OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

                for (BufferedImage frame : frames) {
                    Mat source = bufferedImageToMat(frame);
                    Mat destination = new Mat();
                    cvtColor(source, destination, COLOR_GRAY2BGR);
                    Frame convertedFrame = converter.convert(destination);
                    recorder.record(convertedFrame);
                }

                recorder.stop();
            } catch (Exception e) {
                e.printStackTrace();
            }
    }

    public static float sigmoid(float x) {
        return 1.0f / (1.0f + (float)Math.exp(-x));
    }

    public static void main(String[] args) throws OrtException, java.io.IOException, java.lang.InterruptedException {
        // Get the full paths of the FITS files
        Vector<String> fitsFiles = ReadFITS.read_dir(args[1]);

        // Load the ONNX Model
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        try (OrtSession.SessionOptions opts = new SessionOptions()) {
            opts.setOptimizationLevel(OptLevel.BASIC_OPT);
            logger.info("Loading model from " + args[0]);

            // Open up an ONNX Runtime Session.
            try (OrtSession session = env.createSession(args[0], opts)) {

                logger.info("Inputs:");
                for (NodeInfo i : session.getInputInfo().values()) {
                    logger.info(i.toString());
                }

                logger.info("Outputs:");
                for (NodeInfo i : session.getOutputInfo().values()) {
                    logger.info(i.toString());
                }
      
                // Now loop through all the FITS files, building up the queue till
                // it reaches 16 - the window size for this model.

                Vector<float[][]> queue = new Vector<>();
                float[][][][][] stack = new float[1][1][WINDOW_LENGTH][IMG_HEIGHT][IMG_WIDTH]; // B,C,D,H,W
                String inputName = session.getInputNames().iterator().next();
                assert fitsFiles.size() > WINDOW_LENGTH;
                Vector<BufferedImage> final_frames = new Vector<>();
                
                // This line isn't necessary really but it cuts down on the number of fits we want to process.
                List<String> subfiles = fitsFiles.subList(0, 32);

                for (String file_name : subfiles) {
                    byte[][] raw_img = ReadFITS.read(file_name);
                    float[][] final_img = ReadFITS.prepare_fits(raw_img);
                    queue.add(final_img);

                    if (queue.size() > WINDOW_LENGTH) {
                        queue.remove(0);

                        for (int i = 0; i < WINDOW_LENGTH; i++) {
                            float[][] frame = queue.get(i);
                            stack[0][0][i] = frame;
                        }

                        OnnxTensor stack_data = OnnxTensor.createTensor(env, stack);
                        Result output = session.run(Collections.singletonMap(inputName, stack_data));
                        OnnxValue val = output.get(0);

                        if (val.getType() == OnnxValue.OnnxValueType.ONNX_TYPE_TENSOR) {
                            OnnxTensor pred_tensor = (OnnxTensor)val;
                            FloatBuffer fb = pred_tensor.getFloatBuffer();

                            // Save our output frame - the last one from the output.
                            int num_pixels = IMG_HEIGHT * IMG_WIDTH;
                            float[][] pred_frame = new float[IMG_HEIGHT][IMG_WIDTH];

                            // Move to the last frame position
                            int last_frame_pos = IMG_HEIGHT * IMG_WIDTH * (WINDOW_LENGTH-1);
                            fb = fb.position(last_frame_pos);
        
                            for (int i = 0; i < num_pixels; i++) {
                                float value = fb.get();
                                // Make sure to pass through the sigmoid confidence value.
                                value = sigmoid(value);

                                if (value >= CONFIDENCE) {
                                    value = 1.0f;
                                } else {
                                    value = 0.0f;
                                }

                                int y = i / IMG_WIDTH;
                                int x = i % IMG_WIDTH;
                                pred_frame[y][x] = value;
                            }

                            // Convert frame to a buffered image.
                            BufferedImage bi_frame = convertToBufferedImage(pred_frame);
                            //File outputfile = new File("saved.png");
                            //ImageIO.write(bi_frame, "png", outputfile);
                            final_frames.add(bi_frame);
                        }
                    }
                }

                convertToWebM(final_frames, "prediction.webm");
            }   
        }
    }
}
