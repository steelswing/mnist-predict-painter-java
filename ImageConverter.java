
import java.awt.image.BufferedImage;
import java.io.IOException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * File: ImageConverter.java
 * Created on 8 авг. 2023 г., 05:38:27
 *
 * @author LWJGL2
 */
public class ImageConverter {

    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private static final int CHANNELS = 1; // Для черно-белых изображений

    public static INDArray convertToINDArray(BufferedImage image) throws IOException {
        INDArray array = Nd4j.create(1, HEIGHT * WIDTH * CHANNELS);
        INDArray singleData = array.getRow(0);

        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                int pixel = image.getRGB(x, y);

                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = (pixel >> 0) & 0xFF;

                singleData.putScalar(y * image.getHeight() + x, r / 255.0);
            }
        }
        return array;
    }
}
