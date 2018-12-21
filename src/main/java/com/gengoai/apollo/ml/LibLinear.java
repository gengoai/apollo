package com.gengoai.apollo.ml;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.*;
import de.bwaldvogel.liblinear.Model;

import java.util.Iterator;

/**
 * <p>Helper functions for working with LibLinear</p>
 *
 * @author David B. Bracewell
 */
public final class LibLinear {

   private LibLinear() {
      throw new IllegalAccessError();
   }


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


   /**
    * Estimates an outcome for the given data using the given LibLinearModel
    *
    * @param model     the model
    * @param data      the data
    * @param biasIndex the bias index
    * @return the nd array
    */
   public static NDArray estimate(Model model, NDArray data, int biasIndex) {
      double[] p = new double[model.getNrClass()];
      if (model.isProbabilityModel()) {
         Linear.predictProbability(model, LibLinear.toFeature(data, biasIndex), p);
      } else {
         Linear.predictValues(model, LibLinear.toFeature(data, biasIndex), p);
      }
      //re-arrange the probabilities to match the target feature
      double[] prime = new double[model.getNrClass()];
      int[] labels = model.getLabels();
      for (int i = 0; i < labels.length; i++) {
         prime[labels[i]] = p[i];
      }
      return NDArrayFactory.rowVector(prime);
   }

   /**
    * Estimates an outcome for the given data using the given LibLinearModel
    *
    * @param model     the model
    * @param data      the data
    * @param biasIndex the bias index
    * @return the nd array
    */
   public static double regress(Model model, NDArray data, int biasIndex) {
      return Linear.predict(model, toFeature(data, biasIndex));
   }

   /**
    * Fits a LibLinear model given a a data supplier and set of parameters.
    *
    * @param dataSupplier the data supplier
    * @param parameter    the parameter
    * @param verbose      the verbose
    * @param numFeatures  the num features
    * @param biasIndex    the bias index (-1 if no bias)
    * @return the model
    */
   public static Model fit(SerializableSupplier<MStream<NDArray>> dataSupplier,
                           Parameter parameter,
                           boolean verbose,
                           int numFeatures,
                           int biasIndex
                          ) {
      Problem problem = new Problem();
      problem.l = (int) dataSupplier.get().count();
      problem.x = new Feature[problem.l][];
      problem.y = new double[problem.l];
      problem.n = numFeatures + 1;
      problem.bias = biasIndex >= 0 ? 0 : -1;
      dataSupplier.get().zipWithIndex()
                  .forEach((datum, index) -> {
                     problem.x[index.intValue()] = toFeature(datum, biasIndex);
                     problem.y[index.intValue()] = datum.getLabelAsDouble();
                  });

      if (verbose) {
         Linear.enableDebugOutput();
      } else {
         Linear.disableDebugOutput();
      }

      return Linear.train(problem, parameter);
   }

}//END OF LibLinear
