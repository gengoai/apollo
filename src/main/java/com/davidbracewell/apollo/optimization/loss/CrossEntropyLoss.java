package com.davidbracewell.apollo.optimization.loss;

import com.davidbracewell.Math2;
import com.davidbracewell.apollo.linalg.Vector;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;
import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class CrossEntropyLoss implements LossFunction, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public double derivative(double predictedValue, double trueValue) {
      return predictedValue - trueValue;
   }

   @Override
   public double loss(double predictedValue, double trueValue) {
      return trueValue * FastMath.log(predictedValue);
   }

   @Override
   public double loss(Vector predictedValue, Vector trueValue) {
      double loss = 0;
      for (Iterator<Vector.Entry> itr = trueValue.nonZeroIterator(); itr.hasNext(); ) {
         Vector.Entry e = itr.next();
         loss += e.getValue() * Math2.safeLog(predictedValue.get(e.getIndex()));
      }
      return -loss;
   }

}// END OF CrossEntropyLoss
