package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.optimization.DecayLearningRate;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.Weights;
import com.davidbracewell.apollo.optimization.activation.SigmoidActivation;
import com.davidbracewell.apollo.optimization.loss.LogLoss;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author David B. Bracewell
 */
public class DBNLearner extends ClassifierLearner {
   @Getter
   @Setter
   private int[] hiddenLayerSizes = {300};
   @Getter
   @Setter
   @Builder.Default
   private int maxPreTrainIterations = 100;
   @Getter
   @Setter
   @Builder.Default
   private int preTrainBatchSize = 20;
   @Getter
   @Setter
   @Builder.Default
   private LearningRate preTrainLearningRate = new DecayLearningRate(0.1, 0.001);

   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      BernoulliRBM[] rbms = new BernoulliRBM[hiddenLayerSizes.length + 1];
      FeedForwardLearner.FeedForwardLearnerBuilder builder = FeedForwardLearner.builder();
      for (int i = 0; i < hiddenLayerSizes.length; i++) {
         int inputSize = (i == 0)
                         ? dataset.getFeatureEncoder().size()
                         : rbms[i - 1].getNumHidden();
         int outputSize = hiddenLayerSizes[i];
         if (i + 1 < hiddenLayerSizes.length) {
            builder.layer(new DenseLayer(outputSize));
            builder.layer(new ActivationLayer(SigmoidActivation.INSTANCE));
         }
         rbms[i] = new BernoulliRBM(outputSize, inputSize);
      }
      rbms[rbms.length - 1] = new BernoulliRBM(rbms[rbms.length - 2].nV, dataset.getLabelEncoder().size());
      builder.outputActivation(SigmoidActivation.INSTANCE);
      builder.lossFunction(new LogLoss());
      //Pretrain
      List<Vector> data = dataset.asVectors().collect();
      for (int i = 0; i < maxPreTrainIterations; i++) {
         Collections.shuffle(data);
         for (Vector datum : data) {
            Vector[] X = new Vector[rbms.length];
            for (int li = 0; li < rbms.length; li++) {
               Vector x = (li == 0) ? datum : X[li - 1];

               X[li] = SigmoidActivation.INSTANCE.apply(rbms[li].runHiddenProbs(x))
                                                 .map(d -> d >= Math.random() ? 1.0 : 0.0)
                                                 .slice(1);

               rbms[i].train(Arrays.asList(x), 1);
            }

         }
      }

      AtomicInteger index = new AtomicInteger(0);
      builder.layerProcessor(layer -> {
         if (layer.hasWeights()) {
            Matrix W = rbms[index.get()].W;
            Matrix Wcopy = DenseMatrix.zeroes(layer.getOutputSize(), layer.getInputSize());
            for (int r = 1; r < W.numberOfRows(); r++) {
               Wcopy.setRow(r - 1, W.row(r).slice(1, W.row(r).dimension()));
            }
            layer.setWeights(new Weights(Wcopy, W.column(0).slice(1, W.numberOfRows()).copy(), false));
            index.addAndGet(1);
         }
      });
      FeedForwardLearner learner = builder.build();
      return learner.train(dataset);
   }
}// END OF DBNLearner
