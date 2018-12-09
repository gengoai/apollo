package com.gengoai.apollo.ml.classification;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.logging.Loggable;
import com.gengoai.stream.MStream;
import de.bwaldvogel.liblinear.*;

import java.util.Iterator;

/**
 * The type Lib linear model.
 *
 * @author David B. Bracewell
 */
public class LibLinearModel implements Classifier, Loggable {
   private static final long serialVersionUID = 1L;
   private int biasIndex = -1;
   private Model model;


   /**
    * Converts an Apollo vector into an array of LibLinear feature nodes
    *
    * @param vector    the vector to convert
    * @param biasIndex the index of the bias variable (<0 for no bias)
    * @return the feature node array
    */
   public static Feature[] toFeature(NDArray vector, int biasIndex) {
      int size = (int) vector.size() + (biasIndex > 0 ? 1 : 0);
      final Feature[] feature = new Feature[size];
      int index = 0;
      for (Iterator<NDArray.Entry> itr = vector.sparseOrderedIterator(); itr.hasNext(); index++) {
         NDArray.Entry entry = itr.next();
         feature[index] = new FeatureNode(entry.matrixIndex() + 1, entry.getValue());
      }
      if (biasIndex > 0) {
         feature[size - 1] = new FeatureNode(biasIndex, 1.0);
      }
      return feature;
   }

   @Override
   public Classifier copy() {
      LibLinearModel copy = new LibLinearModel();
      copy.biasIndex = biasIndex;
      copy.model = Copyable.copy(model);
      return copy;
   }

   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      Problem problem = new Problem();
      problem.l = (int) dataSupplier.get().count();
      problem.x = new Feature[problem.l][];
      problem.y = new double[problem.l];
      problem.n = fitParameters.numFeatures + 1;
      problem.bias = fitParameters.bias ? 0 : -1;
      biasIndex = (fitParameters.bias ? fitParameters.numFeatures + 1 : -1);

      dataSupplier.get().zipWithIndex()
                  .forEach((datum, index) -> {
                     problem.x[index.intValue()] = toFeature(datum, biasIndex);
                     problem.y[index.intValue()] = datum.getLabelAsDouble();
                  });

      if (fitParameters.verbose) {
         Linear.enableDebugOutput();
      } else {
         Linear.disableDebugOutput();
      }

      model = Linear.train(problem, new Parameter(fitParameters.solver, fitParameters.C, fitParameters.eps));
   }

   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters parameters) {
      fit(dataSupplier, Cast.as(parameters, Parameters.class));
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   @Override
   public NDArray estimate(NDArray data) {
      double[] p = new double[model.getNrClass()];
      if (model.isProbabilityModel()) {
         Linear.predictProbability(model, toFeature(data, biasIndex), p);
      } else {
         Linear.predictValues(model, toFeature(data, biasIndex), p);
      }

      //re-arrange the probabilities to match the target feature
      double[] prime = new double[model.getNrClass()];
      int[] labels = model.getLabels();
      for (int i = 0; i < labels.length; i++) {
         prime[labels[i]] = p[i];
      }

      data.setPredicted(NDArrayFactory.rowVector(prime));
      return data;
   }

   /**
    * The type Parameters.
    */
   public static class Parameters extends FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The C.
       */
      public double C = 1.0;
      /**
       * The Bias.
       */
      public boolean bias = false;
      /**
       * The Eps.
       */
      public double eps = 0.0001;
      /**
       * The Solver.
       */
      public SolverType solver = SolverType.L2R_LR;
   }
}//END OF LibLinearModel
