package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.analysis.Optimum;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.stream.MStream;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public abstract class GradientOptimizer implements Optimizer {
   final Optimizable.ByGradient function;
   final MStream<Vector> data;

   protected GradientOptimizer(Optimizable.ByGradient function, MStream<Vector> data) {
      this.function = function;
      this.data = data;
   }

   @Override
   public final Optimum getGoal() {
      return Optimum.MINIMUM;
   }

   @Override
   public Vector optimize() {
      while (true) {
         Vector gradient = SparseVector.zeros(function.getParameters().dimension());
         double value = 0;
         for (Iterator<Vector> itr = data.iterator(); itr.hasNext(); ) {
            Vector example = itr.next();
            function.iterate(example);
            value += function.getValue();
            gradient.addSelf(function.getGradient());
         }
         function.getParameters().multiplySelf(gradient);
         if (value == 0) {
            break;
         }
      }
      return function.getParameters();
   }


}// END OF GradientOptimizer
