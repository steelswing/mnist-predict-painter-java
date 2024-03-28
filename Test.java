import java.io.File;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * File: Test.java
 * Created on 8 авг. 2023 г., 04:48:04
 *
 * @author LWJGL2
 */
public class Test {

    private static final double LEARNING_RATE = 0.0003;
    private static final double GRADIENT_THRESHOLD = 100.0;
    private static final IUpdater UPDATER = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();

    public static void main(String[] args) throws Throwable {
        final int height = 28;
        final int width = 28;
        final int channels = 1; // MNIST grayscale images have only 1 channel
        final int outputNum = 10; // Number of output classes (digits 0-9)
        final int batchSize = 100;
        final int nEpochs = 100000;

        // Create a dataset iterator for MNIST
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 42);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 42);

        // Build the neural network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(UPDATER)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .nOut(28)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(128)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .build();

        // Create the neural network
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        // Print the model summary
        System.out.println(model.summary());

        model.setListeners(new PerformanceListener(100, true));

        SimpleMNISTPainter painter = new SimpleMNISTPainter(model);
        painter.updateVisual(null);
        mnistTrain.reset();
        // Train the model
        int iter = 0;
        for (int i = 0; i < nEpochs; i++) {
            System.out.println("Run epoch: " + i);
            if (i == 4) {
                System.out.println("Save model.....");
                model.save(new File("saved_model.zip"));
            }
            while (mnistTrain.hasNext()) {
                iter++;
                DataSet next = mnistTrain.next();

                model.fit(next);
                if (iter % 20 == 0) {
                    painter.updateVisual(null);
                    mnistTest.reset();
                }

            }
            mnistTrain.reset();
        }
    }

    public static void main1(String[] args) throws Throwable {
        System.out.println("Loading model....");
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("saved_model.zip"), true);
        System.out.println(model.summary());

        SimpleMNISTPainter painter = new SimpleMNISTPainter(model);
        painter.updateVisual(null);

    }
}
