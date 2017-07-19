package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;

import java.io.Serializable;

/**
 * The interface Weight initializer.
 *
 * @author David B. Bracewell
 */
@FunctionalInterface
public interface WeightInitializer extends Serializable {

   /**
    * The constant DEFAULT.
    */
   WeightInitializer DEFAULT = (m) -> {
      double max = Math.sqrt(6.0) / Math.sqrt(m.getNumberOfFeatures() + m.getNumberOfLabels());
      double min = -max;
      for (int r = 0; r < m.numberOfWeightVectors(); r++) {
         for (int c = 0; c < m.getNumberOfFeatures(); c++) {
            m.getWeightVector(r).set(c, min + (max - min) * Math.random());
         }
      }
      return m;
   };

   WeightInitializer ZEROES = (m) -> {
      for (int i = 0; i < m.numberOfWeightVectors(); i++) {
         Vector v = m.getWeightVector(i);
         v.nonZeroIterator().forEachRemaining(e -> v.set(e.index, 0));
      }
      return m;
   };

   /**
    * Initialize.
    *
    * @param weights the weights
    */
   WeightMatrix initialize(WeightMatrix weights);


}// END OF WeightInitializer
