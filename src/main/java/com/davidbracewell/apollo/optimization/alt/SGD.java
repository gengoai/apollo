package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.ConstantLearningRate;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.guava.common.base.Stopwatch;
import com.davidbracewell.stream.MStream;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class SGD implements Optimizer {


   public static Vector optimize(Vector theta, List<Vector> data) {
      LearningRate learningRate = new ConstantLearningRate(1.0);
      LossFunction lossFunction = new LogLoss();
      TerminationCriteria tc = TerminationCriteria.create()
                                                  .maxIterations(200)
                                                  .tolerance(1e-4)
                                                  .historySize(3);
      int numProcessed = 0;
      double bias = 0;
      double lr = learningRate.getInitialRate();
      for (int iteration = 0; iteration < tc.maxIterations(); iteration++) {
         double totalLoss = 0;
         Stopwatch sw = Stopwatch.createStarted();
//         int batchSize = data.size() / 20;
//         for (int batch = 0; batch < data.size(); batch += batchSize) {
//            double totalError = 0;
//            Vector gradient = Vector.sZeros(theta.dimension());
//            for (int i = batch; i < batch + batchSize && i < data.size(); i++) {
//               double predicted = Activation.SIGMOID.apply(theta.dot(data.get(i)) + bias);
//               double error = lossFunction.derivative(predicted, data.get(i).getLabelAsDouble());
//               totalLoss += lossFunction.loss(predicted, data.get(i).getLabelAsDouble());
//               gradient.addSelf(data.get(i).mapMultiply(error));
//               totalError += error;
//               numProcessed++;
//            }
//            totalError /= batchSize;
//            theta.subtractSelf(gradient.mapMultiply(lr / batchSize));
//            bias -= totalError * lr;
//            lr = learningRate.get(lr, iteration, numProcessed);
//         }
         for (Vector datum : data) {
            double predicted = Activation.SIGMOID.apply(theta.dot(datum) + bias);
            double error = lossFunction.derivative(predicted, datum.getLabelAsDouble());
            totalLoss += lossFunction.loss(predicted, datum.getLabelAsDouble());
            theta.subtractSelf(datum.mapMultiply(error * lr));
            bias -= error * lr;
            numProcessed++;
            lr = learningRate.get(lr, iteration, numProcessed);
         }
         sw.stop();
         System.out.println("iteration=" + iteration + ", totalLoss=" + totalLoss + ", time=" + sw);
         if (tc.check(totalLoss)) {
            break;
         }
      }


      return theta.insert(0, bias);
   }

   @Override
   public CostWeightTuple optimize(WeightVector theta,
                                   MStream<? extends Vector> stream,
                                   CostFunction costFunction,
                                   TerminationCriteria terminationCriteria,
                                   LearningRate learningRate,
                                   WeightUpdate weightUpdater,
                                   boolean verbose
                                  ) {
      int numProcessed = 0;

      double lr = learningRate.getInitialRate();
      double lastLoss = 0d;
      for (int iteration = 0; iteration < terminationCriteria.maxIterations(); iteration++) {
         Stopwatch sw = Stopwatch.createStarted();
         double totalLoss = 0;
         for (Vector datum : stream.shuffle()) {
            CostGradientTuple cgt = costFunction.evaluate(datum, theta);
            totalLoss += weightUpdater.update(theta, cgt.getGradient(), lr);
            numProcessed++;
            lr = learningRate.get(lr, iteration, numProcessed);
         }
         sw.stop();
         System.out.println("iteration=" + (iteration + 1) + ", totalLoss=" + totalLoss + ", time=" + sw);
         lastLoss = totalLoss;
         if (terminationCriteria.check(totalLoss)) {
            break;
         }
      }
      return CostWeightTuple.of(lastLoss, theta);
   }
}// END OF Optimizer
