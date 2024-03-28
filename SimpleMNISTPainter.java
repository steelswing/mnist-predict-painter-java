import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JFrame;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * File: SimpleMNISTPainter.java
 * Created on 8 авг. 2023 г., 05:28:07
 *
 * @author LWJGL2
 */
public class SimpleMNISTPainter extends JFrame {

    private BufferedImage image;
    private Graphics2D g2d;
    private int prevX, prevY;

    private static final int WIDTH = 28;
    private static final int HEIGHT = 28;
    private static final int SCALE_FACTOR = 20; // Масштаб для увеличения изображения

    private final MultiLayerNetwork model;

    private String result = "";

    public SimpleMNISTPainter(MultiLayerNetwork model) throws InterruptedException {
        this.model = model;
        setTitle("deeplearning4j test ЗАТО НЕ НА ПИТУХОНЕ MNIST Painter");
        setSize(WIDTH * SCALE_FACTOR + 2, HEIGHT * SCALE_FACTOR + 2); // +2 для учета рамки
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        image = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        g2d = image.createGraphics();
        g2d.setColor(Color.BLACK);
        g2d.fillRect(0, 0, WIDTH, HEIGHT);

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                prevX = e.getX() / SCALE_FACTOR;
                prevY = e.getY() / SCALE_FACTOR;

                if (e.getButton() == 3) {
                    g2d.setColor(Color.BLACK);
                    g2d.fillRect(0, 0, WIDTH, HEIGHT);

                    updateVisual((INDArray) null);
                }
            }
        });

        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                int x = e.getX() / SCALE_FACTOR;
                int y = e.getY() / SCALE_FACTOR;
                g2d.setColor(Color.WHITE);
                g2d.drawLine(prevX, prevY, x, y);
                prevX = x;
                prevY = y;
                updateVisual((INDArray) null);

                draw();
            }

        });
        setVisible(true);
        createBufferStrategy(2);

    }

    public static class Result implements Comparable<Result> {

        public int index;
        public double value;

        public Result(int index, double value) {
            this.index = index;
            this.value = value;
        }

        @Override
        public String toString() {
            return String.valueOf(index) + " - " + Math.round(value * 100.0) + "%";
        }

        @Override
        public int compareTo(Result o) {
            return Double.compare(o.value, value);
        }
    }

    protected void updateVisual(INDArray features) {
        try {

            if (features != null) {
                INDArray singleData = features.getRow(0);
                for (int x = 0; x < image.getWidth(); x++) {
                    for (int y = 0; y < image.getHeight(); y++) {
                        int data = (int) (singleData.getDouble(y * image.getHeight() + x) * 255);
                        if (data < 0) {
                            data = 0;
                        }
                        if (data > 255) {

                            data = 255;
                        }
                        image.setRGB(x, y, new Color(data, data, data, 255).getRGB());
                    }
                }

                INDArray vector = model.output(features).getRow(0);
                result = "";
                for (int i = 0; i < 10; i++) {
                    result += String.valueOf(i) + " - " + Math.round(vector.getDouble(i) * 100.0) + "%\n";
                }
            } else {
                INDArray data = ImageConverter.convertToINDArray(image);
                INDArray vector = model.output(data).getRow(0);
                result = "";

                List<Result> resultList = new ArrayList<>();
                for (int i = 0; i < 10; i++) {
                    resultList.add(new Result(i, vector.getDouble(i)));
                }

                resultList.sort(Result::compareTo);
                for (int i = 0; i < resultList.size(); i++) {
                    Result resultValue = resultList.get(i);
                    result += resultValue.toString() + "\n";
                }
            }

            draw();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void draw() {
        final BufferStrategy bufferStrategy = getBufferStrategy();
        Graphics g = bufferStrategy.getDrawGraphics();
        Font font = new Font("Arial", Font.PLAIN, 20);

        g.setColor(Color.BLACK);
        g.fillRect(0, 0, getWidth(), getHeight());

        g.setFont(font);
        g.drawImage(image, 1, 1, WIDTH * SCALE_FACTOR, HEIGHT * SCALE_FACTOR, null);

        // Рисуем сетку
        g.setColor(Color.DARK_GRAY);
        for (int x = 0; x <= WIDTH * SCALE_FACTOR; x += SCALE_FACTOR) {
            g.drawLine(x, 0, x, HEIGHT * SCALE_FACTOR);
        }
        for (int y = 0; y <= HEIGHT * SCALE_FACTOR; y += SCALE_FACTOR) {
            g.drawLine(0, y, WIDTH * SCALE_FACTOR, y);
        }

        g.setColor(Color.red);
        drawStringWithNewLines(g, 10, 50, result);

        g.dispose();
        bufferStrategy.show();
    }



    private void drawStringWithNewLines(Graphics g, int x, int y, String text) {
        FontMetrics fontMetrics = g.getFontMetrics();
        int lineHeight = fontMetrics.getHeight();
        String[] lines = text.split("\n");
        for (String line : lines) {
            g.drawString(line, x, y);
            y += lineHeight;
        }
    }

    public BufferedImage getImage() {
        return image;
    }
}
